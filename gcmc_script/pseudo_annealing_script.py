#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 2022

@author: yun
@purpose: Run sgmc at fixed chemical potential and several temperatures.
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

    def __init__(self, machine='savio', dmus = None, savepath=None, savename=None,
                 ce_file_path='', ensemble_file_path='',
                 temperature=None, discard=0, thin_by=10,
                 saved_occu='/global/scratch/users/yychoi94/CaNaVP/data/855_4525_300_occu.npy'):
        """
        Args:
            self.dmus: tuple. First one of tuple is Ca chemical potential.
        """

        self.machine = machine
        self.dmus = dmus
        if self.dmus is None:
            self.dmus = (-8.55, -4.525)
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
        if temperature is None:
            temperature = [5000, 300, 5000, 300, 5000, 300]
        self.temperature = temperature
        self.discard = discard
        self.thin_by = thin_by
        # This only fit to saved ensemble, final_canvp_ensemble.mson
        # Becareful and check always this fit or not.
        self.flip_table = np.array([[-1,  0,  1,  0,  0,  -1,  1,  0],
                                    [-1,  0,  1,  0,  0,  0,  -1,  1],
                                    [ 0, -1,  1,  0,  0,  -1,  0,  1],
                                    [-2,  1,  1,  0,  0,   0,  0,  0]])
        self.saved_occu = saved_occu

    def running(self):

        start = time.time()
        ensemble = loadfn(self.ensemble_file_path)
        if self.saved_occu:
            init_occu = np.load(self.saved_occu)
        else:
            raise ValueError("Need initial occupancy.")
        end = time.time()
        print(f"{end - start}s for initialization.\n")

        # Set chemical potentials
        chemical_potentials = {'Na+': self.dmus[1], 'Ca2+': self.dmus[0], 'Vacancy': 0, 'V3+': 0, 'V4+': 0,
                               'V5+': 0}
        ensemble.chemical_potentials = chemical_potentials

        # Initializing sampler.
        sampler = Sampler.from_ensemble(ensemble, step_type="tableflip", optimize_basis=False,
                                        flip_table=self.flip_table)
        print(f"Sampling information: {sampler.samples.metadata}\n")
        sampler.anneal(self.temperature, 3000000, init_occu, thin_by=self.thin_by, progress=False)

        # Update flip reactions.
        bits = sampler.mckernels[0].mcusher.bits
        flip_table = sampler.mckernels[0].mcusher.flip_table
        flip_reaction = [flip_vec_to_reaction(u, bits) for u in flip_table]
        sampler.samples.metadata['flip_reaction'] = flip_reaction

        # Saving. TODO: Use flush to backend and do not call sampler everytime.
        filename = "{}_{}_cn_sgmc.mson".format(self.dmus[0], self.dmus[1])
        filepath = self.savepath.replace("test_samples.mson", filename)
        sampler.samples.to_hdf5(filepath)
        # dumpfn(sampler.samples, filepath)
        print("Ca: {}, Na: {} is done. Check {}\n".format(self.dmus[0], self.dmus[1], filepath))

        return


def main(ca_dmu=None, na_dmu=None):
    ce_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_canvp_ce.mson'
    ensemble_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_canvp_ensemble.mson'
    sc_matrix = np.array([[3, 0, 0],
                          [0, 4, 0],
                          [0, 0, 5]])
    discard, thin_by = 0, 10
    temperature = 300

    runner = cnsgmcRunner(dmus=(ca_dmu, na_dmu),
                          ce_file_path=ce_file_path,
                          ensemble_file_path=ensemble_file_path)
    runner.running()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ca_dmu', nargs="+", type=float, required=False, default=None,
                        help="Ca chemical potentials.")
    parser.add_argument('--na_dmu', nargs="+", type=float, required=False, default=None,
                        help="Na chemical potentials.")
    args = parser.parse_args()
    main(args.ca_dmu, args.na_dmu)