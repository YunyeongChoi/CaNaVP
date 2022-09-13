#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2022

@author: yun
@purpose: Base job script to run and save sgmc.
"""

import os
import time
import json
import random
import numpy as np
from copy import deepcopy
from smol.io import load_work
from smol.moca import Sampler
from smol.cofe.space import Vacancy
from smol.moca.sampler.mcusher import Tableflip
from pymatgen.core.sites import Species
from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations \
import (OxidationStateDecorationTransformation, \
        OrderDisorderedStructureTransformation)

from src.setter import PoscarGen

# Input - sc_matrix, chemical potential
# Output - save sample after 5,000,000 runs.
# Expected Errors - different table flip.

MACHINE = "yun"


def running(ca_amt=1/3, na_amt=1/6, ca_dmu=None, na_dmu=None, savepath=None, savename=None,
            ensemble_file_path='', temperature=300, thin_by=10):

    if na_dmu is None:
        na_dmu = [-6]
    if ca_dmu is None:
        ca_dmu = [-6]
    if savepath is None:
        if MACHINE == "yun":
            savepath = '/Users/yun/Desktop/github_codes/CaNaVP/data'
        elif MACHINE == "savio":
            savepath = '/global/scratch/users/yychoi94/CaNaVP'
        else:
            raise ValueError("Check Machine option.")
        if not os.path.exists(savepath):
            raise ValueError("Check directory.")
        else:
            if savename is None:
                savepath = os.path.join(savepath, "test_samples.mson")
            else:
                savepath = os.path.join(savepath, savename)

    start = time.time()
    ensemble = loadfn(ensemble_file_path)
    init_occu, charge = initialized_structure()

    # Initializing sampler.
    sampler = Sampler.from_ensemble(ensemble, step_type="tableflip",
                                    temperature=temperature, optimize_basis=True)
    print(f"Sampling information: {sampler.samples.metadata}\n")
    end = time.time()
    print(f"{end - start}s for initialization.\n")

    for i in ca_dmu:
        for j in na_dmu:
            chemical_potentials = {'Na+': j, 'Ca2+': i, 'Vacancy': 0, 'V3+': 0, 'V4+': 0,
                                   'V5+': 0}
            ensemble.chemical_potentials = chemical_potentials
            sampler.run(10000, init_occu, thin_by=thin_by, progress=True)
            sampler.samples.metadata['flip_reaction'] = \
                sampler.mckernels[0].mcusher._compspace.flip_reactions

            filename = "{}_{}_cn_sgmc.mson".format(i, j)
            filepath = savepath.replace("test_samples.mson", filename)
            dumpfn(sampler.samples, filepath)
            print("Ca: {}, Na: {} is done. Check {}\n".format(i, j, filepath))

    return


def initialized_structure(ce_file_path, ca_amt=1.5, na_amt=0):

    expansion = load_work(ce_file_path)['ClusterExpansion']
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
    groups = poscar_generator.get_ordered_structure(ca_amt, na_amt)
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


def main():

    ce_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/0728_preliminary_ce/0728_canvp_ce' \
                   '.mson '
    ensemble_file_path = '/global/scratch/users/yychoi94/CaNaVP/data/0728_preliminary_ce/' \
                         '0825_0728_canvp_ensemble.mson'
    sc_matrix = np.array([[3, 0, 0],
                          [0, 4, 0],
                          [0, 0, 5]])
    discard, thin_by = 50, 10
    temperature = 300

    running(0.5, 0.5, ensemble_file_path=ensemble_file_path)

    return


if __name__ == '__main__':

    main()


