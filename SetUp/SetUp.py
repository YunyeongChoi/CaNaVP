#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05/03/2022

@author: yun
"""
# General import
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

class nasiconSite(PeriodicSite):
    '''
    Inherit Pymatgen.core.sites.PeriodicSite class.
    Adding neighbor, site information in base NASICON lattice.
    '''
    Basestructuredir = os.path.join(os.getcwd(), "Na4V2(PO4)3_POSCAR")

    def __init__(self, site):
        
        super().__init__(site.species, site.frac_coords, site.lattice)
        self.basestructure = Structure.from_file(nasiconSite.Basestructuredir)
        self.neighbor = self.find_neigbors()
        self.isbsite = self.classify_site()

    def find_neigbors(self) -> list:

        neighborlist = []
        if self.specie.name == "Na":
            for iteration in self.basestructure:
                if 3 < iteration.distance(self) < 3.4 and iteration.specie.name == "Na":
                    neighborlist.append(iteration)
        else:
            neighborlist = None

        return neighborlist

    def get_neighbors(self) -> list:

        return self.neighbor

    def classify_site(self):
        
        if self.specie.name != "Na":
            return None

        elif len(self.neighbor) == 2 and self.specie.name == "Na":
            return False

        elif len(self.neighbor) == 6 and self.specie.name == "Na":
            return True

        else:
            warnings.warn("Site is not classifies well - check data", DeprecationWarning)
            
class nasiconStructure(Structure):
    '''
    Inherite pymatgen.core.structure.Structure class.
    classfy cations in structures to 6b, 18e sites. 
    Each site is nasiconSite object.
    * test classification
    '''
    Basestructuredir = os.path.join(os.getcwd(), "Na4V2(PO4)3_POSCAR")
    
    def __init__(self, structure):
        
        super().__init__(structure.lattice, structure.species, structure.frac_coords,
                         structure.charge, False, False, False, None)
        
        self.bsite = []
        self.esite = []
        self.vsite = []
        self.basestructure = Structure.from_file(nasiconSite.Basestructuredir)
        self.update_sites(structure)
        self.classify_sites()
        
    def update_sites(self, structure):
        
        a_ratio = int(structure.lattice.a / self.basestructure.lattice.a)
        b_ratio = int(structure.lattice.b / self.basestructure.lattice.b)
        c_ratio = int(structure.lattice.c / self.basestructure.lattice.c)

        self.basestructure.make_supercell([a_ratio, b_ratio, c_ratio])
        
        # Need to update supercell case for nasiconSite class
        for i, j in enumerate(self.basestructure):
            nasiconinfo = nasiconSite(j)
            self.basestructure[i] = nasiconinfo
        
        # Updating PerodicSite objects to nasiconSite objects.
        for i, site in enumerate(structure):
            for j, base_site in enumerate(self.basestructure):
                if np.allclose(site.frac_coords, base_site.frac_coords):
                    self[i] = self.basestructure[j]
        
    def classify_sites(self):
        
        # Updating bsite, esite information.
        for i, j in enumerate(self):
            if j.isbsite:
                self.bsite.append(i)
            elif j.isbsite == None and j.specie.name == 'V':
                self.vsite.append(i)
            elif j.isbsite == None:
                pass
            elif not j.isbsite:
                self.esite.append(i)
            else:
                warnings.warn("Not well classified sites", DeprecationWarning)


class poscarGen(object):
    '''
    POSCAR generator including Ground / HE State based on Ewald energy.
    SAVE_DIR?
    Should make structure info file (json) at the end of directory.
    Include b site information, neighbors.
    '''
    Basestructuredir = os.path.join(os.getcwd(), "Na4V2(PO4)3_POSCAR")
    poscar_dir = os.path.join(os.getcwd())

    def __init__(self, calc_dir = "/Users/yun/Desktop/github_codes/CaNaVP/SetUp/calc_test"):
        
        self.basestructure = Structure.from_file(poscarGen.Basestructuredir)
        self.decoratedstructure = nasiconStructure(self.basestructure)
        self.calc_dir = calc_dir

    def get_disordered_structure(self, x, y) -> Structure:
        '''
        Count number of Ca and Na first.
        If Ca + Na > bsite, put Ca first in bsite, then put remainings to esite
        Else put all cations to bsite.
        Here, do not consider Na3 concentration which Na occupation is same in b, e site.
        '''
        structure = deepcopy(self.basestructure)
        V_state = 9 / 2 - x - y / 2

        if 3 < V_state < 4:
            v_site_occs = {'Mn' : V_state - 3, 'Ni' : 4 - V_state}
        elif 4 < V_state < 5:
            v_site_occs = {'Cr' : V_state - 4, 'Mn' : 5 - V_state}
        elif V_state == 4:
            v_site_occs = {'Mn' : 1}
        elif V_state == 3:
            v_site_occs = {'Ni' : 1}
        else:
            warnings.warn("This exceed oxidation state limit of V", DeprecationWarning)

        for i in self.decoratedstructure.vsite:
            structure.replace(i, v_site_occs)

        if x + y < 1:
            b_site_occs = {'Ca' : x, 'Na' : y}
            e_site_occs = {'Ca' : 0, 'Na' : 0}
        else:
            if x < 1:
                b_site_occs = {'Ca' : x, 'Na' : 1 - x}
                e_site_occs = {'Na' : (x + y - 1) / 3}
            else:
                b_site_occs = {'Ca' : 1}
                e_site_occs = {'Ca' : (x - 1) / 3, 'Na' : y / 3}

        for i in self.decoratedstructure.bsite:
            structure.replace(i, b_site_occs)
        for j in self.decoratedstructure.esite:
            structure.replace(j, e_site_occs)

        sites2remove = []
        for k, l in enumerate(structure):
            if l.species.elements == []:
                sites2remove.append(k)

        structure.remove_sites(sites2remove)

        return structure
    
    def get_ordered_structure(self, x, y) -> list:
        
        # Set oxidation States for atoms.
        ox_states = {'Ca' : 2,
                     'Na' : 1,
                     'P' : 5,
                     'Ni' : 3,
                     'Mn' : 4,
                     'Cr' : 5,
                     'O' : -2}
        
        s = self.get_disordered_structure(x, y)
        print("Oxidation State Decorating...")
        osdt = OxidationStateDecorationTransformation(ox_states)
        s = osdt.apply_transformation(s)

        try:
            print("Transforming to Ordered structures...")
            odst = OrderDisorderedStructureTransformation(0, False, False)
            s_list = odst.apply_transformation(s, 5)
        except IndexError:
            # Not a disordered structure, Only one option
            print("Warning - Check structure before you proceed.")
            return [[s]]
        except ValueError:
            print('Warning - empty structure, Handle manually')
            print(x, y)

        print("Structure Matching...")
        matcher = StructureMatcher()
        groups = matcher.group_structures([d['structure'] for d in s_list])
        
        return groups

    def run(self) -> None:
        
        # Generate calculation directory.
        if not os.path.exists(self.calc_dir):
            os.mkdir(self.calc_dir)

        count_folder = 0
        count_structure = 0

        for y in np.arange(0, 3.01, 1 / 3):
            for x in np.arange(0, 1.51 - y / 2, 1 / 6):

                count_folder += 1
                
                if count_folder > 200:
                    continue
                else:
                    # Generate folders
                    foo = self.calc_dir + '/' + str(np.round(x,3)) + '_' + str(np.round(y,3))
                    if not os.path.exists(foo):
                        os.mkdir(foo)
                    print("")
                    print(str(count_folder) + ': ' + foo)
                    
                    groups = self.get_ordered_structure(x, y)
        
                    for i in range(len(groups)):
                        if i == 5:
                            break
                        else:
                            bar = os.path.join(foo, str(i))
                            if not os.path.exists(bar):
                                os.mkdir(bar)
                            dpos = os.path.join(bar, 'POSCAR')
                            if 'Ni3+' in groups[i][0].composition.as_dict().keys():
                                groups[i][0].replace_species({'Ni3+' : 'V3+'})
                            if 'Mn4+' in groups[i][0].composition.as_dict().keys():
                                groups[i][0].replace_species({'Mn4+' : 'V4+'})
                            if 'Cr5+' in groups[i][0].composition.as_dict().keys():
                                groups[i][0].replace_species({'Cr5+' : 'V5+'})
                            #print(groups[i][0].composition.as_dict().keys())
                            groups[i][0].to(fmt='poscar', filename=dpos)
                            count_structure += 1
                            
        print("total {} folders, {} structures are generated.".format(count_folder, count_structure))

        return
    
    def makeHEstste(self) -> Structure:
        '''
        Start from ground structure calculated in get_ordered_structure,
        Force move one Ca or one Na from 6b site to random 18e site.
        '''
        test_poscar = "/Users/yun/Desktop/github_codes/CaNaVP/SetUp/calc/0.167_2.0/0/POSCAR"
        test_structure = Structure.from_file(test_poscar)

        return

class inputGen():
    
    '''
    INCAR, KPOINTS, POTCAR, job script generator for VASP run.
    '''

    def __init__(self):

        return

    
def main():

    return

if __name__ == '__main__':
    main()