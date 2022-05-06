#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05/03/2022

@author: yun
"""
from glob import glob
from copy import deepcopy
from subprocess import call
import os
import warnings
import numpy as np

# Pymatgen import
from pymatgen.symmetry.analyzer import SpacegroupOperations
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.vasp.inputs import Poscar, Kpoints, Incar, Potcar
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core.composition import Composition
from pymatgen.io.vasp import Vasprun
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.standard_transformations import (
OrderDisorderedStructureTransformation, RemoveSpeciesTransformation,      
PartialRemoveSpecieTransformation, OxidationStateDecorationTransformation, 
SubstitutionTransformation)

# Compmatscipy import https://github.com/CJBartel/compmatscipy
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.handy_functions import read_json, write_json, is_slurm_job_in_queue
from compmatscipy.HelpWithVASP import VASPSetUp, VASPBasicAnalysis, JobSubmission, magnetic_els

class NASICONSite(PeriodicSite):
    '''
    Inherit Pymatgen.core.sites.PeriodicSite class.
    Adding neighbor, site information in base NASICON lattice.
    '''
    Basestructuredir = os.path.join(os.getcwd(), "Na4V2(PO4)3_POSCAR")

    def __init__(self, site):
        
        super().__init__(site.species, site.frac_coords, site.lattice)
        self.basestructure = Structure.from_file(NASICONSite.Basestructuredir)
        self.neighbor = self.findneigbors()
        self.isbsite = self.classifysite()

    def findneigbors(self):

        neighborlist = []
        for iteration in self.basestructure:
            if 3 < iteration.distance(self) < 3.4 and iteration.specie.name == 'Na':
                neighborlist.append(iteration)

        return neighborlist

    def getneighbors(self):

        return self.neighbor

    def classifysite(self):

        if len(self.neighbor) == 2:
            return False

        elif len(self.neighbor) == 6:
            return True

        else:
            warnings.warn("Site is not classifies well - check data", DeprecationWarning)


class Structuregen():
    '''
    POSCAR generator including Ground / HE State based on Ewald energy
    SAVE_DIR?
    '''

    def __init__(self):

        return
    
    def cationOcc(self, x, y, b_site_occs, c_site_occs, b_sites_to_modify, c_sites_to_modify):
        
        if option == 'stable':
            if x > 0 and y >= 1:
                b_site_occs = {'Ca' : x, 'Na' : 1 - x}
                c_site_occs = {'Na' : (x + y - 1) / 3}
                for i in b_sites_to_modify:
                    structure.replace(i, b_site_occs)
                for j in c_sites_to_modify:
                    structure.replace(j, c_site_occs)
            elif x == 0 and y != 1:
                b_site_occs = {'Na' : y / 4}
                c_site_occs = {'Na' : y / 4}
                for i in b_sites_to_modify:
                    structure.replace(i, b_site_occs)
                for j in c_sites_to_modify:
                    structure.replace(j, c_site_occs)
            elif x == 0 and y == 1:
                b_site_occs = {'Na' : 1}           
                for i in b_sites_to_modify:
                    structure.replace(i, b_site_occs)
                structure.remove_sites(c_sites_to_modify)
            elif x <= 1 - y and y  < 1:
                b_site_occs = {'Ca' : x, 'Na' : y}
                for i in b_sites_to_modify:
                    structure.replace(i, b_site_occs)
                structure.remove_sites(c_sites_to_modify)
            elif 1 - y < x < 1 and y < 1:
                b_site_occs = {'Ca' : x, 'Na' : 1 - x}
                c_site_occs = {'Na' : (x + y - 1) / 3}
                for i in b_sites_to_modify:
                    structure.replace(i, b_site_occs)
                for j in c_sites_to_modify:
                    structure.replace(j, c_site_occs)
            elif 1 < x and y < 1:
                b_site_occs = {'Ca' : 1}
                c_site_occs = {'Ca' : (x - 1) / 3, 'Na' : y / 3}
                for i in b_sites_to_modify:
                    structure.replace(i, b_site_occs)
                for j in c_sites_to_modify:
                    structure.replace(j, c_site_occs)
            elif x == 1 and y < 1:
                b_site_occs = {'Ca' : 1}
                c_site_occs = {'Na' : y / 3}
                for i in b_sites_to_modify:
                    structure.replace(i, b_site_occs)
                for j in c_sites_to_modify:
                    structure.replace(j, c_site_occs)
                    
        return

class Inputgen():

    def __init__(self):

        return