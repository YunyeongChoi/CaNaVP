#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 2022

@author: yun
@purpose: Abstract job script to run and save charge neutral semi grand monte carlo jobs.
"""

import os
import numpy as np
from abc import abstractmethod, ABCMeta
from monty.serialization import loadfn
from smol.io import load_work
from smol.moca.ensemble.ensemble import Ensemble
from smol.cofe.expansion import ClusterExpansion
from pymatgen.core.structure import Structure


class gcmcabc(metaclass=ABCMeta):

    def __init__(self,
                 machine,
                 dmus,
                 savepath,
                 savename,
                 ce_file_path,
                 ensemble_file_path,
                 ):
        """
        Args:
            machine: Machine that will run jobs.
            dmus: list(tuple). Chemical potentials of Na, Ca that will be used.
                  First one of tuple is Na chemical potential.
            savepath: Directory want to save hdf5 file.
            savename: Name of hdf5 file to save.
            ce_file_path: Cluster Expansion Object path
            ensemble_file_path: Ensemble Object path
        """
        self.machine = machine
        self.dmus = dmus
        self.savepath = savepath
        self.savename = savename
        self.ce_file_path = ce_file_path
        self.ensemble_file_path = ensemble_file_path

        if self.savepath is None:
            if self.machine == "savio":
                self.savepath = '/global/scratch/users/yychoi94/CaNaVP_gcMC/data'
            else:
                raise ValueError("Check machine or savepath option.")
            if not os.path.exists(self.savepath):
                raise FileNotFoundError("Saving directory is not exists.")
            else:
                if self.savename is None:
                    self.savepath = os.path.join(self.savepath, "test_samples.mson")
                else:
                    self.savepath = os.path.join(self.savepath, self.savename)
        else:
            if self.savename is None:
                self.savepath = os.path.join(self.savepath, "test_samples.mson")
            else:
                self.savepath = os.path.join(self.savepath, self.savename)

        if self.ce_file_path is None:
            self.ce_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_data/ce' \
                                '/final_canvp_ce.mson'

        if self.ensemble_file_path is None:
            self.ensemble_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_data/gcmc' \
                                      '/final_canvp_ensemble.mson'

        # This only fit to saved ensemble, ./data/final_data/gcmc/final_canvp_ensemble.mson
        # The different ensemble object can have different sublattice orderings.
        # Becareful and do sanity check always this fit or not.
        self.flip_table = np.array([[-1,  0,  1,  0,  0,  -1,  1,  0],
                                    [-1,  0,  1,  0,  0,  0,  -1,  1],
                                    [0, -1,  1,  0,  0,  -1,  0,  1],
                                    [-2,  1,  1,  0,  0,   0,  0,  0]])

    @property
    def expansion(self) -> ClusterExpansion:

        return load_work(self.ce_file_path)['ClusterExpansion']

    @property
    def ensemble(self) -> Ensemble:

        return loadfn(self.ensemble_file_path)

    @property
    def supercell_matrix(self) -> np.ndarray:

        return self.ensemble.processor.supercell_matrix

    @property
    def primcell(self) -> Structure:

        return self.expansion.structure

    @abstractmethod
    def sanity_check(self):

        return

    @abstractmethod
    def run(self):

        return

    @abstractmethod
    def initialized_structure(self):

        return