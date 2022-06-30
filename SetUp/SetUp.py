#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05/03/2022

@author: yun
"""
# General import
import os
import random
import warnings
import argparse
import numpy as np
from glob import glob
from copy import deepcopy
from subprocess import call

# Pymatgen import
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation, OxidationStateDecorationTransformation)

# Compmatscipy import https://github.com/CJBartel/compmatscipy
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.handy_functions import read_json, write_json, is_slurm_job_in_queue
from compmatscipy.HelpWithVASP import VASPSetUp, VASPBasicAnalysis, JobSubmission, magnetic_els

__basestructure__ = os.path.join(os.getcwd(), "Na4V2(PO4)3_POSCAR")


class NasiconSite(PeriodicSite):
    """
    Inherit Pymatgen.core.sites.PeriodicSite class.
    Adding neighbor, site information in base NASICON lattice.
    """

    # Basestructuredir = os.path.join(os.getcwd(), "Na4V2(PO4)3_POSCAR")

    def __init__(self, site):

        super().__init__(site.species, site.frac_coords, site.lattice)
        self.basestructure = Structure.from_file(__basestructure__)
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


class NasiconStructure(Structure):
    """
    Inherite pymatgen.core.structure.Structure class.
    classfy cations in structures to 6b, 18e sites.
    Each site is NasiconSite object.
    * test classification
    * Need to clean siteinfo. There'll be cleaner way.
    """

    def __init__(self, structure):

        super().__init__(structure.lattice, structure.species, structure.frac_coords,
                         structure.charge, False, False, False, None)

        self.siteinfo = {'b': {'Ca': {}, 'Na': {}},
                         'e': {'Ca': {}, 'Na': {}},
                         'v': {}}
        self.basestructure = Structure.from_file(__basestructure__)
        self.update_sites(structure)
        # self.classify_sites()

    def update_sites(self, structure):

        a_ratio = int(structure.lattice.a / self.basestructure.lattice.a)
        b_ratio = int(structure.lattice.b / self.basestructure.lattice.b)
        c_ratio = int(structure.lattice.c / self.basestructure.lattice.c)

        self.basestructure.make_supercell([a_ratio, b_ratio, c_ratio])

        # Need to update supercell case for NasiconSite class
        for i, j in enumerate(self.basestructure):
            nasiconinfo = NasiconSite(j)
            self.basestructure[i] = nasiconinfo

        # Updating PerodicSite objects to NasiconSite objects.
        for i, site in enumerate(structure):
            for j, base_site in enumerate(self.basestructure):
                if np.allclose(site.frac_coords, base_site.frac_coords):
                    self[i] = NasiconSite(self[i])
                    if base_site.isbsite:
                        if site.specie.name == 'Ca':
                            self.siteinfo['b']['Ca'][i] = (j, base_site.frac_coords)
                        elif site.specie.name == 'Na':
                            self.siteinfo['b']['Na'][i] = (j, base_site.frac_coords)
                        else:
                            warnings.warn("None-cation element placed in 6b site",
                                          DeprecationWarning)
                    elif base_site.isbsite is None and base_site.specie.name == 'V':
                        self.siteinfo['v'][i] = (j, base_site.frac_coords)
                    elif base_site.isbsite is None:
                        pass
                    elif not base_site.isbsite:
                        if site.specie.name == 'Ca':
                            self.siteinfo['e']['Ca'][i] = (j, base_site.frac_coords)
                        elif site.specie.name == 'Na':
                            self.siteinfo['e']['Na'][i] = (j, base_site.frac_coords)
                        else:
                            warnings.warn("None-cation element placed in 18e site",
                                          DeprecationWarning)
                    else:
                        warnings.warn("Not well classified sites", DeprecationWarning)


class PoscarGen(object):
    """
    POSCAR generator including Ground / HE State based on Ewald energy.
    SAVE_DIR?
    Should make structure info file (json) at the end of directory.
    Include b site information, neighbors.
    """

    # Set oxidation States for atoms.
    ox_states = {'Ca': 2, 'Na': 1, 'P': 5, 'Ni': 3, 'Mn': 4, 'Cr': 5, 'O': -2}

    def __init__(self, calc_dir="/Users/yun/Desktop/github_codes/CaNaVP/SetUp/calc_test"):

        self.basestructure = Structure.from_file(__basestructure__)
        self.decoratedstructure = NasiconStructure(self.basestructure)
        self.calc_dir = calc_dir

    def get_disordered_structure(self, x, y) -> Structure:
        """
        Count number of Ca and Na first.
        If Ca + Na > bsite, put Ca first in bsite, then put remainings to esite
        Else put all cations to bsite.
        Here, do not consider Na3 concentration which Na occupation is same in b, e site.
        """
        structure = deepcopy(self.basestructure)
        v_state = 9 / 2 - x - y / 2

        if 3 < v_state < 4:
            v_site_occs = {'Mn': v_state - 3, 'Ni': 4 - v_state}
        elif 4 < v_state < 5:
            v_site_occs = {'Cr': v_state - 4, 'Mn': 5 - v_state}
        elif v_state == 4:
            v_site_occs = {'Mn': 1}
        elif v_state == 3:
            v_site_occs = {'Ni': 1}
        else:
            warnings.warn("This exceed oxidation state limit of V", DeprecationWarning)
            v_site_occs = {'V': v_state}

        for i in self.decoratedstructure.siteinfo['v'].keys():
            structure.replace(i, v_site_occs)

        if x + y < 1:
            b_site_occs = {'Ca': x, 'Na': y}
            e_site_occs = {'Ca': 0, 'Na': 0}
        else:
            if x < 1:
                b_site_occs = {'Ca': x, 'Na': 1 - x}
                e_site_occs = {'Na': (x + y - 1) / 3}
            else:
                b_site_occs = {'Ca': 1}
                e_site_occs = {'Ca': (x - 1) / 3, 'Na': y / 3}

        for i in list(self.decoratedstructure.siteinfo['b']['Ca'].keys()) + \
                 list(self.decoratedstructure.siteinfo['b']['Na'].keys()):
            structure.replace(i, b_site_occs)
        for j in list(self.decoratedstructure.siteinfo['e']['Ca'].keys()) + \
                 list(self.decoratedstructure.siteinfo['e']['Na'].keys()):
            structure.replace(j, e_site_occs)

        sites2remove = []
        for k, l in enumerate(structure):
            if not l.species.elements:
                sites2remove.append(k)

        structure.remove_sites(sites2remove)

        return structure

    def get_ordered_structure(self, x, y) -> list:

        s = self.get_disordered_structure(x, y)
        print("Oxidation State Decorating...")
        osdt = OxidationStateDecorationTransformation(PoscarGen.ox_states)
        s = osdt.apply_transformation(s)

        try:
            print("Transforming to Ordered structures...")
            odst = OrderDisorderedStructureTransformation(0, False, False)
            # noinspection PyTypeChecker
            s_list = odst.apply_transformation(s, 50)
        except IndexError:
            # Not a disordered structure, Only one option
            print("Warning - Check structure before you proceed.")
            return [[s]]
        except ValueError:
            print('Warning - empty structure, Handle manually')
            print(x, y)
            return []

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

        for y in np.arange(0, 3.01, 1 / 6):
            for x in np.arange(0, 1.51 - y / 2, 1 / 6):

                count_folder += 1

                if count_folder > 200:
                    continue
                else:
                    # Generate folders
                    foo = self.calc_dir + '/' + str(np.round(x, 3)) + '_' + str(np.round(y, 3))
                    if not os.path.exists(foo):
                        os.mkdir(foo)
                    print("")
                    print(str(count_folder) + ': ' + foo)

                    groups = self.get_ordered_structure(x, y)
                    if len(groups) == 0:
                        continue

                    for i in range(len(groups)):
                        if i == 5:
                            break
                        else:
                            bar = os.path.join(foo, str(i))
                            if not os.path.exists(bar):
                                os.mkdir(bar)
                            dpos = os.path.join(bar, 'POSCAR')
                            if 'Ni3+' in groups[i][0].composition.as_dict().keys():
                                groups[i][0].replace_species({'Ni3+': 'V3+'})
                            if 'Mn4+' in groups[i][0].composition.as_dict().keys():
                                groups[i][0].replace_species({'Mn4+': 'V4+'})
                            if 'Cr5+' in groups[i][0].composition.as_dict().keys():
                                groups[i][0].replace_species({'Cr5+': 'V5+'})
                            # print(groups[i][0].composition.as_dict().keys())
                            groups[i][0].to(fmt='poscar', filename=dpos)
                            count_structure += 1

        print("total {} folders, {} structures are generated.".format(count_folder,
                                                                      count_structure))

        return

    def makeHEstatehelper(self, x, y, z, poscar, b_sites_in_poscar, e_candidates, option):
        """
        :param poscar: target structure want to change.
        :param b_sites_in_poscar: b sites in the target structure.
        :param e_candidates: available e sites in the base structure.
        :param option: cations. "Ca" or "Na"
        :return:
        """

        if len(b_sites_in_poscar[option]) > 0:
            new_poscar = deepcopy(poscar)
            rand_remove_position = random.choice(b_sites_in_poscar[option])
            print(rand_remove_position)
            new_poscar.remove_sites([rand_remove_position])
            rand_add_position = random.choice(e_candidates)
            print(rand_add_position)
            new_poscar.append(option,
                              self.decoratedstructure[rand_add_position].coords,
                              True)
            # Sorting
            new_poscar.sort()
            # Saving
            dir_name = os.path.join(self.calc_dir,
                                    str(np.round(x, 3)) + '_' + str(np.round(y, 3)),
                                    str(z) + '_' + option + '_HE')
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            new_poscar_name = os.path.join(dir_name, 'POSCAR')

            # Calculate Ewald energy
            # Not availalbe for now

            new_poscar.to("poscar", new_poscar_name)
        else:
            print("In {},{}, No {} in the structure".format(x, y, option))

    def makeHEstate(self, x, y, z, option="Ca") -> None:
        """
        Start from ground structure calculated in get_ordered_structure,
        Force move one Ca or one Na from 6b site to random 18e site.
        Need all bsite e site information.
        Need to clean a bit. Using too many lists.
        x: Ca concentration (float),
        y: Na concentration (float),
        z: nth order of Ewald energy (int).
        option: Str, 'Ca' or 'Na' to move to 18e from 6b site.
        """

        poscar_dir = os.path.join(self.calc_dir, str(np.round(x, 3)) + '_' + str(np.round(y, 3)),
                                  str(z), 'POSCAR')
        poscar = NasiconStructure(Structure.from_file(poscar_dir))

        # Save e site info in base structure. In base basis.
        e_sites_in_base = list(self.decoratedstructure.siteinfo['e']['Ca'].keys()) \
                          + list(self.decoratedstructure.siteinfo['e']['Na'].keys())

        # Save b, e site info in poscar structure. In poscar basis index.
        e_sites_in_poscar = {'Ca': list(poscar.siteinfo['e']['Ca'].keys()),
                             'Na': list(poscar.siteinfo['e']['Na'].keys())}
        b_sites_in_poscar = {'Ca': list(poscar.siteinfo['b']['Ca'].keys()),
                             'Na': list(poscar.siteinfo['b']['Na'].keys())}

        # Transfer e_sites_in_poscar to base basis index
        e_sites_in_poscar_base_basis = []
        for i in e_sites_in_poscar['Ca']:
            e_sites_in_poscar_base_basis.append(poscar.siteinfo['e']['Ca'][i][0])
        for j in e_sites_in_poscar['Na']:
            e_sites_in_poscar_base_basis.append(poscar.siteinfo['e']['Na'][j][0])

        # Save available e sites in the base structure.
        e_candidates = []

        # Get availalbe e sites in the base structure.
        for i in e_sites_in_base:
            if i not in e_sites_in_poscar_base_basis:
                e_candidates.append(i)

        # Move Ca to esite
        if option == "Ca" or option == "Na":
            self.makeHEstatehelper(x, y, z, poscar, b_sites_in_poscar, e_candidates, option)
        else:
            warnings.warn("{} is not supported".format(option), DeprecationWarning)

        return

    def HEstaterun(self):
        """
        Generate HE states from structures from PoscarGen.run
        """
        spec_list = glob(self.calc_dir + "/*/")
        for i in spec_list:
            detailed_spec_list = glob(i + "*/")
            for j in detailed_spec_list:
                poscar_dir = os.path.join(j, "POSCAR")
                # noinspection PyTypeChecker
                x = float(poscar_dir.split('/')[-3].split('_')[0])
                # noinspection PyTypeChecker
                y = float(poscar_dir.split('/')[-3].split('_')[1])
                z = poscar_dir.split('/')[-2]
                # noinspection PyTypeChecker
                if not ("Ca_HE" in z or "Na_HE" in z):
                    z = int(z)
                    self.makeHEstate(x, y, z, "Ca")
                    self.makeHEstate(x, y, z, "Na")

        return


class InputGen:
    """
    INCAR, KPOINTS, POTCAR, job script generator for VASP run.
    """

    def __init__(self, machine, hpc, calc_dir, convergence_option):
        """
        :param machine: Machine that build files. YUN, cori, savio, stampede, bridges2, ginar.
        :param hpc: Machine that actually run calculations. cori, savio, stampede, bridges2 ginar.
        :param calc_dir: Directory that contains POSCAR file.
        test_calc_dir = '/Users/yun/Desktop/github_codes/CaNaVP/SetUp/calc_test/0.5_1.0'
        test_poscar = '/Users/yun/Desktop/github_codes/CaNaVP/SetUp/calc_test/0.5_1.0/0/POSCAR'
        test_structure = Structure.from_file(test_poscar)

        Need to add option fast / full option.
        Need to remove dependencies.
        Need to write consecutive run from fast to full.
        """
        self.machine = machine
        self.hpc = hpc
        self.calc_dir = calc_dir
        self.convergence_option = convergence_option

        return

    def get_U(self):

        U_els = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9,
                 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2}
        els = VASPSetUp(self.calc_dir).ordered_els_from_poscar()
        U = {el: {'L': 2 if el in U_els.keys() else 0,
                  'U': U_els[el] if el in U_els.keys() else 0} for el in els}
        for el in U:
            if U[el]['U'] != 0:
                return U

        return False

    def get_incar(self, ISIF=3, U=True) -> None:
        """
        :param ISIF: 3 for structural optimization, 2 for NEB
        :param U: +U if U is true.
        :return: Write INCAR in self.calc_dir directory.
        """
        vsu = VASPSetUp(self.calc_dir)

        if self.convergence_option == 'full':
            additional = {'EDIFFG': -1e-2,
                          'EDIFF': 1e-5,
                          'NPAR': 16,
                          'NSW': 600,
                          'IBRION': 2}
        elif self.convergence_option == 'fast':
            additional = {'EDIFFG': -0.05,
                          'EDIFF': 1e-4,
                          'NPAR': 16,
                          'NSW': 300,
                          'IBRION': 2,
                          'ENCUT': 400}
        else:
            additional = {}
            warnings.warn("Check convergence option", DeprecationWarning)

        if ISIF == 3:
            additional['ISIF'] = ISIF
        elif ISIF == 2:
            additional['ISIF'] = ISIF
            additional['ICHARG'] = 0
            additional['ISTART'] = 1
        else:
            warnings.warn("Check ISIF option", DeprecationWarning)

        if U:
            ordered_els = vsu.ordered_els_from_poscar()
            U_data = self.get_U()
            additional['LDAU'] = 'TRUE'
            additional['LDAUTYPE'] = 2
            additional['LDAUPRINT'] = 1
            additional['LDAUJ'] = ' '.join([str(0) for _ in ordered_els])
            additional['LDAUL'] = ' '.join([str(U_data[el]['L']) for el in U_data])
            additional['LDAUU'] = ' '.join([str(U_data[el]['U']) for el in U_data])

        additional['ISYM'] = 0
        additional['ISMEAR'] = 0
        additional['AMIX'] = 0.1
        additional['BMIX'] = 0.01

        vsu.incar(is_geometry_opt=True, mag='fm', additional=additional)

    def get_kpoints(self) -> None:

        fkpoints = os.path.join(self.calc_dir, 'KPOINTS')
        s = Structure.from_file(os.path.join(self.calc_dir, 'POSCAR'))
        if self.convergence_option == 'full':
            Kpoints().automatic_density(s, kppa=1000).write_file(fkpoints)
        else:
            Kpoints().gamma_automatic().write_file(fkpoints)

    def get_potcar(self) -> None:

        obj = VASPSetUp(self.calc_dir)
        obj.potcar(machine=self.machine, MP=True)

    @property
    def get_jobname(self) -> str:

        a = self.calc_dir.split('/')

        return '-'.join(a[-3:-1])

    def get_sub(self):

        nodes = 4
        ntasks = 256
        walltime = '12:00:00'
        err_file = 'log.e'
        out_file = 'log.o'
        options = {'error': err_file,
                   'out': out_file,
                   'time': walltime,
                   'nodes': nodes,
                   'ntasks': ntasks,
                   'job-name': self.get_jobname}

        if self.hpc == 'stampede':

            account = 'TG-DMR970008S'
            partition = 'normal'
            options['account'] = account
            options['partition'] = partition
            launch_line = '    ibrun -n {} /home1/06991/tg862905/bin/' \
                          'vasp.5.4.4_vtst178_with_DnoAugXCMeta/vasp_std > vasp.out\n'.format(
                ntasks)

        elif self.hpc == 'bridges2':

            account = 'dmr060032p'
            partition = 'RM'
            del options['ntasks']
            options['account'] = account
            options['partition'] = partition
            options['ntasks-per-node'] = 128
            launch_line = '    module load intelmpi/20.4-intel20.4\n    mpirun -n {} ' \
                          '/jet/home/yychoi/bin/Bridges2/vasp_gam > vasp.out\n'.format(nodes * 128)

        elif self.hpc == 'savio':

            account = 'co_condoceder'
            qos = 'savio_lowprio'
            partition = 'savio3'
            ntasks = nodes * 32
            options['account'] = account
            options['partition'] = partition
            options['qos'] = qos
            options['ntasks'] = ntasks
            launch_line = '    mpirun -n {} /global/home/users/yychoi94/bin/vasp.5.4' \
                          '.4_vtst178_with_DnoAugXCMeta/vasp_std > vasp.out\n'.format(ntasks)

        elif self.hpc == 'cori':

            account = 'm1268'
            constraint = 'knl'
            qos = 'regular'
            options['account'] = account
            options['constraint'] = constraint
            options['qos'] = qos
            launch_line = '    srun -n {} /global/homes/y/yychoi/bin/VASP_20190930/KNL/vasp.5.4' \
                          '.4_vtst178_with_DnoAugXCMeta/vasp_std_knl > vasp.out\n'.format(ntasks)

        else:
            launch_line = ''
            warnings.warn("Check hps option", DeprecationWarning)

        fsub = os.path.join(self.calc_dir, 'job.sh')

        line1 = '#!/bin/bash\n'
        with open(fsub, 'w') as f:
            f.write(line1)
            for tag in options:
                option = options[tag]
                if option:
                    option = str(option)
                    f.write('%s --%s=%s\n' % ('#SBATCH', tag, option))
            f.write('\n')

            line = 'mkdir U; cd U;\n'
            f.write(line)
            line = "IsConv=`grep 'required accuracy' OUTCAR`;\n"
            f.write(line)
            line = 'if [ -z "${IsConv}" ]; then\n'
            f.write(line)
            line = '    if [ -s "CONTCAR" ]; then cp CONTCAR POSCAR; fi;\n'
            f.write(line)
            line = '    if [ ! -s "POSCAR" ]; then\n'
            f.write(line)
            line = '        cp ../{KPOINTS,POTCAR,POSCAR} .;\n'
            f.write(line)
            line = '    fi\n'
            f.write(line)
            line = '        cp ../INCAR .;\n'
            f.write(line)
            f.write(launch_line)
            line = 'fi'
            f.write(line)
            f.close()

    def at_once(self):

        self.get_incar()
        self.get_kpoints()
        self.get_potcar()
        self.get_sub()

        return


def main(machine, hpc, option, inputoption) -> None:
    calc_dir = ''
    if machine == 'savio':
        calc_dir = '/global/scratch/users/yychoi94/CaNaVP/SetUp'
    elif machine == 'cori':
        calc_dir = '/global/cscratch1/sd/yychoi/JCESR/CaNaVP/SetUp'
    elif machine == 'stampede':
        calc_dir = '/scratch/06991/tg862905/JCESR/CaNaVP/SetUp'
    elif machine == 'YUN':
        calc_dir = '/Users/yun/Desktop/github_codes/CaNaVP/SetUp'
    elif machine == 'bridges2':
        calc_dir = '/ocean/projects/dmr060032p/yychoi/CaNaVP/SetUp'
    else:
        warnings.warn("Check machine option", DeprecationWarning)

    if not os.path.exists(calc_dir):
        warnings.warn("Running in the wrong machine", DeprecationWarning)

    if hpc not in ['savio', 'cori', 'stampede', 'bridges2']:
        warnings.warn("Check hpc option", DeprecationWarning)

    calc_dir = os.path.join(calc_dir, 'calc')

    if not inputoption:
        poscarrun = PoscarGen(calc_dir)
        # Get ground state POSCAR.
        poscarrun.run()

    if not inputoption:
        # Get HE state POSCAR.
        poscarrun.HEstaterun()

    count = 0
    groundcount = 0
    calclist = {}
    groundlist = {}
    # Get setup files for generated folder.
    spec_list = glob(calc_dir + "/*/")
    for i in spec_list:
        detailed_spec_list = glob(i + "*/")
        for j in detailed_spec_list:
            count += 1
            calclist[count] = j
            if str(0) in j.split("/")[-2]:
                groundcount += 1
                groundlist[count] = j
            inputgenerator = InputGen(machine, hpc, j, option)
            inputgenerator.at_once()

    fjson = '/ocean/projects/dmr060032p/yychoi/CaNaVP/SetUp/calc_list.json'
    write_json(calclist, fjson)
    groundjson = '/ocean/projects/dmr060032p/yychoi/CaNaVP/SetUp/ground_list.json'
    write_json(groundlist, groundjson)
    print(count)

    return


def launchjobs() -> None:
    return


def changelatticevector():
    # Will be deleted.
    from pymatgen.core.lattice import Lattice

    test_poscar = "/Users/yun/Desktop/github_codes/CaNaVP/SetUp/calc/0.167_2.0/0/POSCAR"
    test_structure = Structure.from_file(test_poscar)
    test_lattice = Lattice([[16.7288, 0., 0.],
                            [-4.3644, 7.559363, 0.],
                            [0., 0., 21.8042]])

    new_structure = Structure(test_lattice, test_structure.species,
                              test_structure.frac_coords,
                              test_structure.charge, False, False, False, None)

    return new_structure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=True, default='YUN',
                        help="Machine that want to run this python file. Yun, cori, stampede, "
                             "bridges2, savio, done are available.")
    parser.add_argument('-p', type=str, required=True, default='savio',
                        help="HPC that want to run DFT calculation. cori, stampede, "
                             "bridges2, savio, done are available.")
    parser.add_argument('-o', type=str, required=False, default='fast',
                        help="Option for DFT calculation. fast or full.")
    parser.add_argument('-i', type=bool, required=False, default=True,
                        help="Option for input generation. If true, only input generator runs.")
    args = parser.parse_args()

    main(args.m, args.p, args.o, args.i)
