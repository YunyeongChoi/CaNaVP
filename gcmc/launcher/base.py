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


class cnsgmcbase(metaclass=ABCMeta):

    def __init__(self,
                 machine,
                 ca_amt,
                 na_amt,
                 saved_occu,
                 dmus,
                 savepath,
                 savename,
                 ce_file_path,
                 ensemble_file_path,
                 ):
        """
        Args:
            machine: Machine that will run jobs.
            ca_amt: Initial Ca amount. Unit is per V2(PO4)3
            na_amt: Initial Na amount. Unit is per V2(PO4)3
            saved_occu: np.array. Initial occupancy of Structure that want to start from.
            dmus: list(tuple). Chemical potentials of Na, Ca that will be used.
                  First one of tuple is Na chemical potential.
            savepath: Directory want to save hdf5 file.
            savename: Name of hdf5 file to save.
            ce_file_path: Cluster Expansion Object path
            ensemble_file_path: Ensemble Object path
        """
        self.machine = machine
        self.ca_amt = ca_amt
        self.na_amt = na_amt
        self.saved_occu = saved_occu
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

        if self.ce_file_path is None:
            self.ce_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_data/ce' \
                                '/final_canvp_ce.mson '

        if self.ensemble_file_path is None:
            self.ensemble_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/final_data/gcmc' \
                                      '/final_canvp_ensemble.mson'

        # This only fit to saved ensemble, ./data/final_data/gcmc/final_canvp_ensemble.mson
        # The different ensemble object can have different sublattice orderings.
        # Becareful and do sanity check always this fit or not.
        self.flip_table = np.array([[-1,  0,  1,  0,  0,  -1,  1,  0],
                                    [-1,  0,  1,  0,  0,  0,  -1,  1],
                                    [ 0, -1,  1,  0,  0,  -1,  0,  1],
                                    [-2,  1,  1,  0,  0,   0,  0,  0]])

    @abstractmethod
    def sanity_check(self):

        return

    @abstractmethod
    def running(self):

        return

    @abstractmethod
    def initialized_structure(self):

        return

