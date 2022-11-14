#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/28/2022

@author: yun

@purpose: Retrieve convergence, energy, magnetic moment from VASP calculations.
Errors and Wanrings are also collected for consecutive running of VASP.

@development:
1. Need Data cleaning for further cluster expansion and Monte Carlo simulations.
2. Need to parse stress tensor from VASP calculations. Not urgent task.
3. Need to parse CHGCAR..?
4. Need an abstract class 'retriever'. Inherit retriever class to make child classes, vasp_retriever,
   NEB_retriever, AIMD_retriever, SCAN_retriever, vdw_retriever. Currently only vasp_retriever is
   generated using brute force.
5. Need to Decide the best design patterns.
6. Need to develop test.
"""

# General import
import os
import numpy as np
from pathlib import Path

# Pymatgen import
from pymatgen.core.structure import Structure
from dft.vaspexception import NeutralChargeError, AtomMoveError

test_folder = "/Users/yun/Desktop/github_codes/CaNaVP/setup/test_calc_2"


class vasp_retriever(object):

    def __init__(self, calc_dir):
        """
        Args:
            calc_dir: Directory that VASP ran.
        Attribute:
            self.energy: Energy
            self.stress_tensor: Stress tensor
            self.setting: Setting of VASP run from OUTCAR
            self.magmom: Magnetic moments of atoms from OUTCAR
            self.warns: Warnings
            self.errors: Errors
        """

        self.calc_dir = calc_dir
        self.energy = 0
        self.stress_tensor = self.get_stress_tensor()
        self.setting = self.get_setting()
        self.magmom = self.get_magnetic_moment()
        self.warns = {}
        self.errors = []
        self.band = ''
        self.chg = ''
        self.dos = ''

        return

    @staticmethod
    def is_int_float(element):

        try:
            int(element)
            return int(element)
        except ValueError:
            try:
                float(element)
                return float(element)
            except ValueError:
                return element

    def get_setting(self, dimension_params=None,
                    start_params=None,
                    electronic_params_first=None,
                    electronic_params_second=None,
                    ionic_params=None,
                    dos_params=None,
                    band_params=None,
                    write_params=None,
                    dipole_params=None,
                    u_params=None,
                    xc_params=None,
                    lr_params=None,
                    orbital_mag_params=None
                    ) -> dict:
        """
        Collect the all settings from OUTCAR.
        Need update for SCAN, vdw, NEB calculation
        """
        if dimension_params is None:
            dimension_params = {"Dimension of arrays": ['NKPTS', 'NKDIM', 'NBANDS', 'NEDOS',
                                                        'NIONS', 'LDIM', 'LMDIM', 'NPLWV', 'IRMAX',
                                                        'IRDMAX', 'NGX', 'NGY', 'NGZ', 'NGXF',
                                                        'NGYF', 'NGZF']}
        if start_params is None:
            start_params = {"Startparameter for this run": ['NWRITE', 'PREC', 'ISTART', 'ICHARG',
                                                            'ISPIN', 'LNONCOLLINEAR', 'LSORBIT',
                                                            'INIWAV', 'LASPH', 'METAGGA']}
        if electronic_params_first is None:
            electronic_params_first = \
                {"Electronic Relaxation 1": ['ENCUT', 'ENINI', 'ENAUG', 'NELM', 'EDIFF', 'LREAL',
                                             'NLSPLINE', 'LCOMPAT', 'GGA_COMPAT', 'LMAXPAW',
                                             'LMAXMIX', 'VOSKOWN', 'ROPT']}
        if electronic_params_second is None:
            electronic_params_second = \
                {"Electronic relaxation 2 (details)": ['IALGO', 'LDIAG', 'LSUBROT', 'TURBO',
                                                       'IRESTART', 'NREBOOT', 'NMIN', 'EREF',
                                                       'IMIX', 'AMIX', 'BMIX', 'AMIX_MAG',
                                                       'BMIX_MAG', 'AMIN', 'WC', 'INIMIX', 'MIXPRE',
                                                       'MAXMIX']}
        if ionic_params is None:
            ionic_params = {"Ionic relaxation": ['EDIFFG', 'NSW', 'NBLOCK', 'IBRION', 'NFREE',
                                                 'ISIF', 'IWAVPR', 'ISYM', 'LCORR', 'POTIM', 'TEIN',
                                                 'TEBEG', 'SMASS', 'SCALEE', 'NPACO', 'APACO',
                                                 'PSTRESS']}
        if dos_params is None:
            dos_params = {"DOS related values": ['EMIN', 'EMAX', 'EFERMI', 'ISMEAR', 'SIGMA']}
        if band_params is None:
            band_params = {"Intra band minimization": ['WEIMIN', 'EBREAK', 'DEPER', 'TIME']}
        if write_params is None:
            write_params = {"Write flags": ['LWAVE', 'LDOWNSAMPLE', 'LCHARG', 'LVTOT', 'LVHAR',
                                            'LELF', 'LORBIT']}
        if dipole_params is None:
            dipole_params = {"Dipole corrections": ['LMONO', 'LDIPOL', 'IDIPOL', 'EPSILON']}
        if u_params is None:
            u_params = {"LDA+U is selected, type is set to LDAUTYPE": ['LDAUTYPE', 'LDAUL', 'LDAUU',
                                                                       'LDAUJ']}
        if xc_params is None:
            xc_params = {"Exchange correlation treatment:": ['GGA', 'LEXCH', 'VOSKOWN', 'LHFCALC',
                                                             'LHFONE', 'AEXX']}
        if lr_params is None:
            lr_params = {"Linear response parameters": ['LEPSILON', 'LRPA', 'LNABLA', 'LVEL',
                                                        'LINTERFAST', 'KINTER', 'CSHIFT',
                                                        'OMEGAMAX', 'DEG_THRESHOLD', 'RTIME',
                                                        'WPLASMAI', 'DFIELD']}
        if orbital_mag_params is None:
            orbital_mag_params = {"Orbital magnetization related": ['ORBITALMAG', 'LCHIMAG', 'DQ',
                                                                    'LLRAUG']}

        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        params = {}
        for d in [dimension_params, start_params, electronic_params_first, electronic_params_second,
                  ionic_params, dos_params, band_params, write_params, dipole_params, u_params,
                  xc_params, lr_params, orbital_mag_params]:
            params.update(d)
        data = {}

        with open(outcar) as f:
            for line_msg in params:
                for param in params[line_msg]:
                    f.seek(0)
                    line_appeared = False
                    concised = []
                    count = 0
                    for line in f:
                        if line_msg in line:
                            line_appeared = True
                        if line_appeared and (param in line):
                            if param not in ['ROPT', 'LDAUL', 'LDAUU', 'LDAUJ']:
                                val = line.split(param)[1].split('=')[1].strip().split(' ')[0]. \
                                    replace(';', '')
                                data[param] = self.is_int_float(val)
                                break
                            elif param in ['LDAUL', 'LDAUU', 'LDAUJ']:
                                val_list = line.split(param)[1].split('=')[1].strip().split(' ')
                                for i in val_list:
                                    if i != '':
                                        concised.append(self.is_int_float(i))
                                data[param] = concised
                                break
                            else:
                                count += 1
                                val_list = line.split(param)[1].split('=')[1].strip().split(' ')
                                for i in val_list:
                                    if i != '':
                                        concised.append(self.is_int_float(i))
                                data[param] = concised
                                if count == 2:
                                    break
                        else:
                            continue

        return data

    @property
    def is_converged(self) -> bool:

        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        oszicar = os.path.join(self.calc_dir, 'OSZICAR')

        if not os.path.exists(outcar):
            return False
        with open(outcar) as f:
            contents = f.read()
            if 'Elaps' not in contents:
                return False

        nelm, nsw = self.setting['NELM'], self.setting['NSW']
        if nsw == 0:
            max_ionic_step = 1
        else:
            if os.path.exists(oszicar):
                with open(oszicar) as f:
                    for line in f:
                        if 'F' in line:
                            step = line.split('F')[0].strip()
                            if ' ' in step:
                                step = step.split(' ')[-1]
                            step = int(step)
            else:
                with open(outcar) as f:
                    f.seek(0)
                    for line in f:
                        if ('Iteration' in line) and ('(' in line):
                            step = line.split('Iteration')[1].split('(')[0].strip()
                            step = int(step)
            max_ionic_step = step
            if max_ionic_step == nsw:
                return False
        with open(outcar) as f:
            f.seek(0)
            for line in f:
                if ('Iteration' in line) and (str(max_ionic_step) + '(') in line:
                    step = line.split(str(max_ionic_step) + '(')[1].split(')')[0].strip()
                    if int(step) == nelm:
                        return False
        return True

    @property
    def get_energy(self) -> float:

        if not self.is_converged:
            print('calcuation is not converged')
            return np.nan

        oszicar = os.path.join(self.calc_dir, 'OSZICAR')
        if os.path.exists(oszicar):
            with open(oszicar) as f:
                for line in f:
                    if 'F' in line:
                        E = float(line.split('F=')[1].split('E0')[0].strip())
        else:
            outcar = os.path.join(self.calc_dir, 'OUTCAR')
            with open(outcar) as f:
                for line in reversed(list(f)):
                    if 'TOTEN' in line:
                        line = line.split('=')[1]
                        E = float(''.join([c for c in line if c not in [' ', '\n', 'e', 'V']]))
                        break
        return float(E) / self.setting['NIONS']

    @property
    def get_contcar(self) -> Structure:

        fcontcar = os.path.join(self.calc_dir, 'CONTCAR')
        if not os.path.exists(fcontcar):
            raise ValueError("No CONTCAR file exists in directory")
        else:
            contcar = Structure.from_file(fcontcar)

        return contcar

    @property
    def get_poscar(self) -> Structure:

        upper_calc_dir = str(Path(self.calc_dir).parents[0])
        fposcar = os.path.join(upper_calc_dir, 'POSCAR')
        if not os.path.exists(fposcar):
            raise ValueError("No POSCAR file exists in directory")
        else:
            poscar = Structure.from_file(fposcar)

        return poscar

    def check_atom_movement(self) -> None:

        # Check POSCAR / CONTCAR difference.
        # POSCAR should be original one. Not after the continuous runs.

        poscar = self.get_poscar
        contcar = self.get_contcar

        for i, j in enumerate(poscar):
            poscar_cart_coords = j.coords
            contcar_cart_coords = contcar[i].coords
            distance = np.linalg.norm(poscar_cart_coords, contcar_cart_coords)

            # Maybe need warning class?
            try:
                if distance > 1.5:
                    raise AtomMoveError("{}th atom moved it's position".format(i))
                    # warnings.warn("{}th atom moved it's position".format(i), DeprecationWarning)
            except AtomMoveError:
                self.errors.append("{}th Atom-Moved".format(i))

        return

    def get_magnetic_moment(self) -> list:
        """
        :return: List of total absolute magnetic moment in order of structure.
        """

        if not self.is_converged:
            return []

        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        count = 0
        recount = 0
        magcount = 0
        magmoms = []

        with open(outcar, 'r') as f:
            for line in f:
                if "magnetization (x)" in line:
                    count += 1

        with open(outcar, 'r') as g:
            for line in g:
                if "magnetization (x)" in line:
                    recount += 1
                if count == recount:
                    magcount += 1
                    if magcount > 4:
                        magmoms.append(abs(float(line.split(" ")[-1].strip())))
                    if magcount == self.setting['NIONS'] + 4:
                        break

        if not (len(magmoms) == self.setting['NIONS']):
            raise ValueError("Number of Magnetic moments are not same as number of ions.")

        return magmoms

    def get_oxidation_decorated_structure(self):
        """
        check folder - error occured during charge assignment.
        /global/scratch/users/yychoi94/CaNaVP/setup/calc/1.0_0.5/0_Ca_HE/U
        Only applied for CaNaVP for now.
        :return: Oxidation decorated structure based on the magmoms.
        """

        if not self.is_converged:
            return self.get_contcar

        structure = self.get_contcar
        mag = self.magmom

        ox_ranges = {'V': {(0, 0.2): 5, (0.8, 1.2): 4, (1.6, 2.0): 3}}
        default_ox = {'Ca': 2, 'Na': 1, 'P': 5, 'O': -2}
        ox_list = []
        solved_ox = True
        for site_i, site in enumerate(structure.sites):
            assigned = False
            if site.species_string in ox_ranges.keys():
                for (minmag, maxmag), magox in ox_ranges[site.species_string].items():
                    if minmag <= mag[site_i] < maxmag:
                        ox_list.append(magox)
                        assigned = True
                        break

            elif site.species_string in default_ox.keys():
                ox_list.append(default_ox[site.species_string])
                assigned = True
            if not assigned:
                solved_ox = False

        if solved_ox:
            structure.add_oxidation_state_by_site(ox_list)
            try:
                if np.abs(structure.charge) > 0.1:
                    raise NeutralChargeError("Charge is wrongly assigned.")
            except NeutralChargeError:
                # May has better way.
                self.errors.append("Non-neutral")

        # Need to add non-solved ox error.

        return structure

    def get_stress_tensor(self) -> dict:

        # To track stability of cathode.
        # Need to develop.

        return {}
