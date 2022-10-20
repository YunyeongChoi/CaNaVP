#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2022

@author: yun
@purpose: Base job script to run and save sgmc.
"""

import os
import time
import argparse
import numpy as np
from copy import deepcopy
from smol.io import load_work
from smol.moca import Sampler
from monty.serialization import dumpfn, loadfn
from src.setter import PoscarGen
from pymatgen.core.structure import Structure
from gcmc_script.gcmc_utils import get_dim_ids_by_sublattice, flip_vec_to_reaction


class cnsgmcRunner:

    def __init__(self, machine='savio', ca_amt=0.5, na_amt=1.0, dmus = None,
                 savepath=None, savename=None, ce_file_path='', ensemble_file_path='',
                 temperature=300, discard=0, thin_by=10,
                 struct_address='/global/scratch/users/yychoi94/CaNaVP/data/83_44_5000_unsorted_POSCAR'):
        """
        Args:
            self.dmus: list(tuple). First one of tuple is Na chemical potential.
        """

        self.machine = machine
        self.ca_amt = ca_amt
        self.na_amt = na_amt
        self.dmus = dmus
        if self.dmus is None:
            self.dmus = [(-6.0, -6.0), (-6.0, -5.0)]
        self.savename = savename
        self.savepath = savepath
        if self.savepath is None:
            if self.machine == "yun":
                self.savepath = '/Users/yun/Desktop/github_codes/CaNaVP/data'
            elif self.machine == "savio":
                self.savepath = '/global/scratch/users/yychoi94/CaNaVP_gcMC/data'
            else:
                raise ValueError("Check Machine option.")
            if not os.path.exists(self.savepath):
                raise ValueError("Check directory.")
            else:
                if self.savename is None:
                    self.savepath = os.path.join(self.savepath, "test_samples.mson")
                else:
                    self.savepath = os.path.join(self.savepath, self.savename)
        self.ce_file_path = ce_file_path
        self.ensemble_file_path = ensemble_file_path
        self.temperature = temperature
        self.discard = discard
        self.thin_by = thin_by
        # This only fit to saved ensemble, final_canvp_ensemble.mson
        # Becareful and check always this fit or not.
        self.flip_table = np.array([[-1,  0,  1,  0,  0,  -1,  1,  0],
                                    [-1,  0,  1,  0,  0,  0,  -1,  1],
                                    [ 0, -1,  1,  0,  0,  -1,  0,  1],
                                    [-2,  1,  1,  0,  0,   0,  0,  0]])
        self.struct_address = struct_address

    def running(self):

        start = time.time()
        ensemble = loadfn(self.ensemble_file_path)
        if self.struct_address:
            init_struct = Structure.from_file(self.struct_address)
        else:
            init_struct = self.initialized_structure()
        init_occu = ensemble.processor.occupancy_from_structure(init_struct)
        end = time.time()
        print(f"{end - start}s for initialization.\n")

        for i in self.dmus:
            # Set chemical potentials
            chemical_potentials = {'Na+': i[1], 'Ca2+': i[0], 'Vacancy': 0, 'V3+': 0, 'V4+': 0,
                                   'V5+': 0}
            ensemble.chemical_potentials = chemical_potentials

            # Initializing sampler.
            sampler = Sampler.from_ensemble(ensemble, step_type="tableflip",
                                            temperature=self.temperature, optimize_basis=False,
                                            flip_table=self.flip_table)
            print(f"Sampling information: {sampler.samples.metadata}\n")
            sampler.run(5000000, init_occu, thin_by=self.thin_by, progress=False)

            # Update flip reactions.
            bits = sampler.mckernels[0].mcusher.bits
            flip_table = sampler.mckernels[0].mcusher.flip_table
            flip_reaction = [flip_vec_to_reaction(u, bits) for u in flip_table]
            sampler.samples.metadata['flip_reaction'] = flip_reaction

            # Saving. TODO: Use flush to backend and do not call sampler everytime.
            filename = "{}_{}_cn_sgmc.mson".format(i[0], i[1])
            filepath = self.savepath.replace("test_samples.mson", filename)
            sampler.samples.to_hdf5(filepath)
            # dumpfn(sampler.samples, filepath)
            print("Ca: {}, Na: {} is done. Check {}\n".format(i[0], i[1], filepath))

        return

    def initialized_structure(self):

        expansion = load_work(self.ce_file_path)['ClusterExpansion']
        prim_cell = deepcopy(expansion.structure)
        intermediate_sc_matrix = np.array([[3, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])
        prim_cell.make_supercell(intermediate_sc_matrix)
        # remake prim_cell to available basestructure.
        for i, j in enumerate(prim_cell):
            if j.species_string == 'Na+:0.333, Ca2+:0.333':
                prim_cell.replace(i, 'Na')
            elif j.species_string == 'V3+:0.333, V4+:0.333, V5+:0.333':
                prim_cell.replace(i, 'V')
        # Generate ground state.
        poscar_generator = PoscarGen(basestructure=prim_cell)
        groups = poscar_generator.get_ordered_structure(self.ca_amt, self.na_amt)
        # Retrieve V TM.
        if 'Ni3+' in groups[0][0].composition.as_dict().keys():
            groups[0][0].replace_species({'Ni3+': 'V3+'})
        if 'Mn4+' in groups[0][0].composition.as_dict().keys():
            groups[0][0].replace_species({'Mn4+': 'V4+'})
        if 'Cr5+' in groups[0][0].composition.as_dict().keys():
            groups[0][0].replace_species({'Cr5+': 'V5+'})

        final_supercell = groups[0][0]
        remain_sc_matrix = np.array([[1, 0, 0],
                                     [0, 4, 0],
                                     [0, 0, 5]])
        final_supercell.make_supercell(remain_sc_matrix)

        return final_supercell


def main(ca_amt=0.5, na_amt=1.0, ca_dmu=None, na_dmu=None):
    ce_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_canvp_ce.mson'
    ensemble_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_canvp_ensemble.mson'
    sc_matrix = np.array([[3, 0, 0],
                          [0, 4, 0],
                          [0, 0, 5]])
    discard, thin_by = 0, 10
    temperature = 300

    # Handling input string list to float list
    ca_dmu_float = []
    for i in ca_dmu:
        ca_dmu_float.append(float(''.join(i)))
    na_dmu_float = []
    for i in na_dmu:
        na_dmu_float.append(float(''.join(i)))

    dmus = []
    if not len(ca_dmu_float) == len(na_dmu_float):
        raise ValueError("Cannot couple chemical potentials. Check input.")
    for i, j in enumerate(ca_dmu_float):
        dmus.append((float(j), float(na_dmu_float[i])))

    runner = cnsgmcRunner(ca_amt=ca_amt, na_amt=na_amt, dmus=dmus,
                          ce_file_path=ce_file_path, ensemble_file_path=ensemble_file_path)
    runner.running()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ca_amt', type=float, required=False, default=0.5,
                        help="Amount of Ca in initial structure.")
    parser.add_argument('--na_amt', type=float, required=False, default=1.0,
                        help="Amount of Na in initial structure.")
    parser.add_argument('--ca_dmu', nargs="+", type=list, required=False, default=None,
                        help="List of Ca chemical potentials.")
    parser.add_argument('--na_dmu', nargs="+", type=list, required=False, default=None,
                        help="List of Na chemiocal potentials.")
    args = parser.parse_args()
    main(args.ca_amt, args.na_amt, args.ca_dmu, args.na_dmu)
