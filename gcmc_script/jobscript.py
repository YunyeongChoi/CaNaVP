#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2022

@author: yun
@purpose: Base job script to run and save sgmc.
"""


import time
import json
import random
import numpy as np
from copy import deepcopy
from smol.io import load_work
from smol.cofe.space import Vacancy
from smol.moca.sampler.mcusher import Tableflip
from pymatgen.core.sites import Species
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations \
import (OxidationStateDecorationTransformation, \
        OrderDisorderedStructureTransformation)


class job:

