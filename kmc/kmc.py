import os
import sys
import json
import time
import decimal
import random
import pickle
import argparse
import pymatgen
import numpy as np
import warnings
from functools import wraps
from random import sample
from copy import deepcopy
from collections import defaultdict
from itertools import combinations
from pymatgen.core import Species, Composition
from pymatgen.analysis.local_env import BrunnerNN_real

from smol.io import load_work
from smol.moca import Sampler
from smol.cofe import ClusterExpansion
from smol.moca import CompositeProcessor, ClusterExpansionProcessor, EwaldProcessor, Ensemble
import smol.cofe.space.domain as ForVac

from monty.serialization import loadfn

np.set_printoptions(threshold=sys.maxsize)

sys.path.append('../')

simple_barrier_key = {
    'Na': 0.400,
    'Ca': 0.600,
    'Ca_Ca': 0.600,
    'Ca_Na': 0.500,
    'Na_Na': 0.400,
    'Na_Ca': 0.500
}


# Utils
def checktime(func):
    @wraps(func)
    def checktime_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return checktime_wrapper


def read_json(fjson):
    with open(fjson) as f:
        return json.load(f)


def write_json(d, fjson):
    with open(fjson, 'w') as f:
        json.dump(d, f, cls=NpEncoder)
    return d


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class kmcNasicon:

    def __init__(self,
                 ce_file_path,
                 ensemble_file_path,
                 nnmatrix_path,
                 occu_path):

        self.ce = loadfn(ce_file_path)
        self.ensemble = loadfn(ensemble_file_path)
        self.NNmatrix = self.getNNmatrix(nnmatrix_path)
        self.processor = self.ensemble.processor
        self.coef = self.ensemble.processor.coefs
        if ".npz" in occu_path:
            data = np.load(occu_path)
            self.occu = data['o'][-1]
        else:
            self.occu = np.load(occu_path)

        # Cations
        self.Ca = Species.from_string("Ca2+")
        self.Na = Species.from_string("Na+")
        self.Vac = ForVac.Vacancy()
        # TMs
        self.V3 = Species.from_string("V3+")
        self.V4 = Species.from_string("V4+")
        self.V5 = Species.from_string("V5+")
        # Anions
        self.P5 = Species.from_string("P5+")
        self.O2 = Species.from_string("O2-")

        # Dictionary to save data.
        self.save_data = {}

    def getPositionEncoding(self, occupancy):
        """
        Return position encoding, which has key as species,
        and value as sites that specie has taken.
        """

        # Site encoding
        position_encoding = {self.Ca: {'6b': [], '18e': [], 'tot': []},
                             self.Na: {'6b': [], '18e': [], 'tot': []},
                             self.Vac: {'6b': [], '18e': [], 'tot': []},
                             self.V3: [],
                             self.V4: [],
                             self.V5: []}
        encoder = {}
        anion_charge = 0

        for ith, sublattice in enumerate(self.processor.get_sublattices()):
            elements = sublattice.site_space.composition.elements

            if elements == [self.Ca, self.Na]:
                decoder = {}
                spec = {}
                for nth, j in enumerate(sublattice.encoding):
                    decoder[j] = sublattice.species[nth]
                    spec[sublattice.species[nth]] = j
                encoder['cat'] = spec
                active_sites = sublattice.active_sites
                for site_index in active_sites:
                    position_encoding[decoder[occupancy[site_index]]]['tot'].append(site_index)

            elif elements == [self.V3, self.V4, self.V5]:
                decoder = {}
                spec = {}
                for nth, j in enumerate(sublattice.encoding):
                    decoder[j] = sublattice.species[nth]
                    spec[sublattice.species[nth]] = j
                encoder['tm'] = spec
                active_sites = sublattice.active_sites
                for site_index in active_sites:
                    position_encoding[decoder[occupancy[site_index]]].append(site_index)

            elif elements == [self.P5]:
                anion_charge += 5 * len(sublattice.sites)

            elif elements == [self.O2]:
                anion_charge -= 2 * len(sublattice.sites)

            else:
                pass

        assert len(position_encoding[self.V3]) * 3 + \
               len(position_encoding[self.V4]) * 4 + \
               len(position_encoding[self.V5]) * 5 + \
               len(position_encoding[self.Ca]['tot']) * 2 + \
               len(position_encoding[self.Na]['tot']) + anion_charge == 0

        return position_encoding, encoder

    def getNNmatrix(self, nnmatrix_data):
        # Need to check following the site order or not.
        # Need to change hard coded part.
        # supercell should be full cation one
        if not os.path.exists(nnmatrix_data):
            NNmatrix = np.empty([1200, 6])
            NN = BrunnerNN_real(tol=3, cutoff=4)  # tol, cutoff is only for Ca,Na-V2(PO4)3
            for i in range(1200):
                data = NN.get_nn_info(self.processor.structure, i)
                candidate_list = []
                for j in data:
                    neighbor = j['site'].species_string
                    if neighbor == 'Na+:0.333, Ca2+:0.333':
                        candidate_list.append(j['site_index'])
                a = np.array(candidate_list)
                if len(a) == 2:
                    a = np.append(a, [-1, -1, -1, -1])
                NNmatrix[i][:] = a

                if i % 24 == 0:
                    print(f"{i // 24}% done")

            np.save(nnmatrix_data, NNmatrix)
            return NNmatrix.astype(int)

        else:
            return np.load(nnmatrix_data)

    def classifySite(self,
                     position_encoding,
                     target: pymatgen.core.periodic_table.Species):
        """
        With Given NNmatrix, classify sites in position_encoding values.
        """

        if target not in position_encoding.keys():
            raise ValueError("That atom is not exists.")

        for i in position_encoding[target]['tot']:
            if len(np.where(self.NNmatrix[i] > -0.1)[0]) == 6:
                position_encoding[target]['6b'].append(i)
            elif len(np.where(self.NNmatrix[i] > -0.1)[0]) == 2:
                position_encoding[target]['18e'].append(i)
            else:
                warnings.warn("misclassification")

        return position_encoding

    def updatePositionEncoding(self,
                               position_encoding):

        position_encoding = self.classifySite(position_encoding, self.Ca)
        position_encoding = self.classifySite(position_encoding, self.Na)
        position_encoding = self.classifySite(position_encoding, self.Vac)

        return position_encoding

    def getSiteEncoding(self, position_encoding):
        """
        Return site_encoding, sitekey_encoding.
        site_encoding: information of each site.
        sitekey_encoding: information of each site key.
        """

        site_encoding = {}
        sitekey_encoding = {'6b': [], '18e': []}
        cation_sites = []
        for site_key in [self.Ca, self.Na, self.Vac]:
            cation_sites += position_encoding[site_key]['tot']
        cation_sites.sort()

        for i in cation_sites:
            for site_key in [self.Ca, self.Na, self.Vac]:
                if i in position_encoding[site_key]['tot']:
                    if i in position_encoding[site_key]['6b']:
                        sitekey_encoding['6b'].append(i)
                        site_encoding[i] = [site_key, '6b']
                    elif i in position_encoding[site_key]['18e']:
                        sitekey_encoding['18e'].append(i)
                        site_encoding[i] = [site_key, '18e']

        sitekey_encoding['6b'].sort()
        sitekey_encoding['18e'].sort()

        return site_encoding, sitekey_encoding

    def farthestNeighbors(self, sitekey_encoding):
        """
        With given NNmatrix, get farthest pair at all 6b sites.
        Maybe this can be saved.
        """
        structure = self.processor.structure
        farthestNeighborInfo = {}
        for site_number in sitekey_encoding['6b']:
            farthestNeighborInfo[site_number] = []
            neighbor = self.NNmatrix[site_number]
            for i in range(len(neighbor)):
                for j in range(i + 1, len(neighbor)):
                    if structure[neighbor[i]].distance(structure[neighbor[j]]) > 6:
                        farthestNeighborInfo[site_number].append([neighbor[i], neighbor[j]])

        return farthestNeighborInfo

    def farthestHelper(self,
                       farthestNeighborInfo, center, near_vacancy):

        info = farthestNeighborInfo[center]
        for i in info:
            if near_vacancy in i:
                result = [x for x in i if x != near_vacancy]

        return result[0]

    def getHopCandidate(self,
                        site_encoding,
                        sitekey_encoding,
                        farthestNeighborInfo):
        """
        1. Check all the filled site. If neighbor is empty, make candidate.
        2. Check all the filled site. If neighbor is empty and it's farthest
        counterpart is filled, make concert hop candidate.
        """

        single_hop_candidates = []
        for key in site_encoding:
            if site_encoding[key][0] == self.Vac:
                continue
            for neighbor in self.NNmatrix[key]:
                if neighbor != -1 and site_encoding[neighbor][0] == self.Vac:
                    single_hop_candidates.append((key, neighbor))

        concert_hop_candidates = []
        for key in site_encoding:
            if site_encoding[key][0] == self.Vac:
                continue
            for neighbor in self.NNmatrix[key]:
                if key in sitekey_encoding['6b']:
                    if site_encoding[neighbor][0] == self.Vac:
                        otherside = self.farthestHelper(farthestNeighborInfo, key, neighbor)
                        if site_encoding[otherside][0] != self.Vac:
                            concert_hop_candidates.append(((key, otherside), (neighbor, key)))
                elif key in sitekey_encoding['18e']:
                    if neighbor != -1 and site_encoding[neighbor][0] == self.Vac:
                        otherside = [x for x in self.NNmatrix[key] if x != -1 and x != neighbor][0]
                        if site_encoding[otherside][0] != self.Vac:
                            concert_hop_candidates.append(((key, otherside), (neighbor, key)))

        return single_hop_candidates, concert_hop_candidates

    def occupancyKernel(self, single_candidates, concert_candidates, site_encoding, encoder):
        """
        Change candidates into list of tuples, considering
        site encodings.
        """
        occupancy_change = {'Ca': [], 'Na': [], 'Ca_Ca': [], 'Ca_Na': [], 'Na_Ca': [], 'Na_Na': []}

        for i in single_candidates:
            change_vector = [(i[0], 2),
                             (i[1], encoder['cat'][site_encoding[i[0]][0]])]
            if change_vector[1][1] == 1:
                occupancy_change['Ca'].append(change_vector)
            elif change_vector[1][1] == 0:
                occupancy_change['Na'].append(change_vector)

        for i in concert_candidates:
            change_vector = [(i[0][1], 2),
                             (i[1][0], encoder['cat'][site_encoding[i[0][0]][0]]),
                             (i[1][1], encoder['cat'][site_encoding[i[0][1]][0]])]
            if change_vector[2][1] == 1 and change_vector[1][1] == 1:
                occupancy_change['Ca_Ca'].append(change_vector)
            elif change_vector[2][1] == 1 and change_vector[1][1] == 0:
                occupancy_change['Ca_Na'].append(change_vector)
            elif change_vector[2][1] == 0 and change_vector[1][1] == 1:
                occupancy_change['Na_Ca'].append(change_vector)
            elif change_vector[2][1] == 0 and change_vector[1][1] == 0:
                occupancy_change['Na_Na'].append(change_vector)

        return occupancy_change

    def getEdiff(self, occu_change, occupancy):

        ediff_data = {}
        for key in occu_change:
            ediff_data[key] = []
            for hop in occu_change[key]:
                ediff = self.processor.coefs @ \
                        self.processor.compute_feature_vector_change(occupancy, hop)
                ediff_data[key].append(ediff)

        return ediff_data

    def barrierKernel(self, ediff, barrier_key):
        """
        Currently set to fixed value. Local CE will be applied.
        get barrier @ getEdiff function.
        Will go into kernel class, inherit KernelABC
        """
        for key in ediff:
            for ith, siteEdiff in enumerate(ediff[key]):
                if siteEdiff > barrier_key[key]:
                    pass
                else:
                    barrier = siteEdiff / 2 + barrier_key[key]
                    if barrier > siteEdiff:
                        ediff[key][ith] = siteEdiff
                    ediff[key][ith] = barrier
        return ediff

    def kmcKernel(self, barrier_dict, hop_dict, t=300):
        """
        launch KMC
        Will go into kernel class, inherit KernelABC
        """
        kB = 8.617e-05
        totlist = []
        lenlist = []
        for key in barrier_dict:
            for barrier in barrier_dict[key]:
                totlist.append(barrier)
            lenlist.append(len(barrier_dict[key]))

        # KMC part
        totlist = np.array(totlist)
        rate_consts = np.exp(-totlist / (kB * t))
        rate_consts_sum = np.sum(rate_consts)
        rate_consts_cumul = [np.sum(rate_consts[0:i + 1]) / rate_consts_sum \
                             for i in range(len(rate_consts))]
        r1 = random.uniform(0, 1)
        hop_index = rate_consts_cumul.index([i for i in rate_consts_cumul \
                                             if i > r1][0])
        r2 = random.uniform(0, 1)
        time_update = decimal.Decimal(-np.log(r2) / rate_consts_sum)

        # Find which hop
        # TODO: prob need sanity check.
        count = 0
        for ith, j in enumerate(lenlist):
            count += j
            if count > hop_index:
                break

        hop_key = list(hop_dict.keys())[ith]
        hop_key_index = hop_index - sum(lenlist[:ith])
        hop_info = hop_dict[hop_key][hop_key_index]
        print(barrier_dict[hop_key][hop_key_index])

        return hop_info, time_update

    def updateKernel(self,
                     hop_info,
                     site_encoding,
                     position_encoding,
                     occupancy):
        """
        update site after KMC run
        Will go into kernel class, inherit KernelABC
        Can be faster if directly track change.
        """
        new_occu = deepcopy(occupancy)

        if len(hop_info) == 2:
            new_occu[hop_info[0][0]] = hop_info[0][1]
            new_occu[hop_info[1][0]] = hop_info[1][1]

        elif len(hop_info) == 3:
            new_occu[hop_info[0][0]] = hop_info[0][1]
            new_occu[hop_info[1][0]] = hop_info[1][1]
            new_occu[hop_info[2][0]] = hop_info[2][1]

        else:
            raise ValueError("wrong hop information")

        position_encoding, encoder = self.getPositionEncoding(new_occu)
        position_encoding = self.updatePositionEncoding(position_encoding)
        site_encoding, sitekey_encoding = self.getSiteEncoding(position_encoding)

        return site_encoding, position_encoding, new_occu

    @checktime
    def aStep(self,
              site_encoding,
              encoder,
              position_encoding,
              sitekey_encoding,
              occupancy,
              farthestNeighborInfo):
        """
        One step for KMC
        processor will go into class variable.
        sitekey_encoding will gointo class variable
        farthestNeighborInfo will go into class variable.
        simple_barrier_key will go into class variable.
        Input site_encoding, position_encoding, sitekey_encoding
        Output is same.
        """
        single_hop_candidates, concert_hop_candidates = self.getHopCandidate(site_encoding,
                                                                             sitekey_encoding,
                                                                             farthestNeighborInfo)
        occu_change = self.occupancyKernel(single_hop_candidates,
                                           concert_hop_candidates,
                                           site_encoding, encoder)
        ediff = self.getEdiff(occu_change, occupancy)
        barrier = self.barrierKernel(ediff, simple_barrier_key)
        hop_info, time_update = self.kmcKernel(barrier, occu_change)
        site_encoding, position_encoding, updated_occupancy = \
            self.updateKernel(hop_info, site_encoding, position_encoding, occupancy)
        step_data = {'occu': updated_occupancy, 'time': float(time_update), 'hop': hop_info}

        return site_encoding, position_encoding, sitekey_encoding, updated_occupancy, step_data

    def run(self,
            nsteps=1000,
            save_path=''):
        """
        Run nsteps of kmc.
        """
        # Initialization.
        position_encoding, encoder = self.getPositionEncoding(self.occu)
        position_encoding = self.updatePositionEncoding(position_encoding)
        site_encoding, sitekey_encoding = self.getSiteEncoding(position_encoding)
        farthestNeighborInfo = self.farthestNeighbors(sitekey_encoding)

        # Run.
        i = 0
        while i < nsteps:
            site_encoding, position_encoding, sitekey_encoding, self.occu, step_data = \
                self.aStep(site_encoding, encoder, position_encoding, sitekey_encoding, self.occu,
                           farthestNeighborInfo)
            self.save_data['Step' + str(i)] = step_data
            i += 1

        write_json(self.save_data, save_path)


def main(occu_data, save_path, nsteps, t):

    ce_file_path = '/scratch/yychoi/CaNaVP/data/final_data/ce' \
                   '/final_canvp_ce.mson'
    ensemble_file_path = '/scratch/yychoi/CaNaVP/data/final_data/gcmc' \
                         '/final_canvp_ensemble_1201.mson'
    nnmatrix_path = '/scratch/yychoi/CaNaVP/data/final_data/gcmc/NNmatrix_1201.npy'

    for ith, j in enumerate(occu_data):

        o = ''.join(j)
        s = ''.join(save_path[ith])
        a = kmcNasicon(ce_file_path, ensemble_file_path, nnmatrix_path, o)
        a.run(nsteps=nsteps, save_path=s)

    return


if __name__ == '__main__':

    """
    ce_file_path = '../CaNaVP/data/final_data/ce/final_canvp_ce.mson'
    ensemble_file_path = '../CaNaVP/data/final_data/gcmc/final_canvp_ensemble_1201.mson'
    test_data = '/Users/yun/Berkeley/Codes/kmc/data/-7.48276_-3.76552_occupancy.npz'
    last_occu_data = './data/-7.5_-3.77.npy'
    nnmatrix_path = './NNmatrix_1201.npy'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--occu', nargs="+", type=list, required=False,
                        default=None, help="Occupancy path list.")
    parser.add_argument('--savepath', nargs="+", type=list, required=False, 
                        default=None, help="Save path list.")
    parser.add_argument('--nsteps', type=int, required=False, default=1000,
                        help="Amount of Na in initial structure.")
    parser.add_argument('--t', type=int, required=False, default=300,
                        help="Number of steps")
    args = parser.parse_args()
    main(args.occu, args.savepath, args.nsteps, args.t)

