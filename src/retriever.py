#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/28/2022

@author: yun
@purpose: retrieve energy, magnetic moment, stress tensor from VASP calculations. Data cleaning for
further cluster expansion.
"""

# General import
import os
import warnings
import numpy as np
from glob import glob
from Src.setter import InputGen

# Pymatgen import
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite

# Compmatscipy import https://github.com/CJBartel/compmatscipy
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.handy_functions import read_json, write_json, is_slurm_job_in_queue
from compmatscipy.HelpWithVASP import VASPSetUp, VASPBasicAnalysis, JobSubmission, magnetic_els


class vasp_retriever(object):

    def __init__(self, calc_dir):

        return

    def get_energy(self):

        return

    def get_magmom(self):

        return

    def get_stress_tensor(self):

        # To track stability of cathode.

        return

