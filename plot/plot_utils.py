#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 2022

@author: yun
@purpose: Plotting gcMC trajectory on the phase diagram.
@description: Using 0728_canvp_ce ECIs, which is centering-L1 fitting.
 Using 0825-0728_canvp_ensemble, which use 3x4x5 supercell, Metropolis-tableflip, 300K semigrand
 ensemble. Ca and Na chemical potentials are changed from -15 to -1.
"""

import os
import numpy as np
import itertools
import matplotlib as mpl
from src.setter import read_json
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.plotting_functions import tableau_colors

FIG_DIR = os.getcwd()
DATA_DIR = FIG_DIR.replace('plot', 'data')
# vasp_data = read_json(os.path.join(DATA_DIR, '0725_ce_fitting_on_preliminary.json'))


def convert_data(d):
    """
    Convert polished raw data to {std_formula (based on CompAnalyzer) : energy per atom}
    """
    converted_data = {}
    for keys in d:
        converted_data[CompAnalyzer(keys).std_formula()] = min(d[keys])

    return converted_data


def triangle_boundary():
    """
    Args:

    Returns:
        the boundary of a triangle where each axis goes 0 to 1
    """
    corners = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
    corners = [triangle_to_square(pt) for pt in corners]
    data = dict(zip(['left', 'top', 'right'], corners))
    lines = {'bottom': {'x': (data['left'][0], data['right'][0]),
                        'y': (data['left'][1], data['right'][1])},
             'left': {'x': (data['left'][0], data['top'][0]),
                      'y': (data['left'][1], data['top'][1])},
             'right': {'x': (data['top'][0], data['right'][0]),
                       'y': (data['top'][1], data['right'][1])}}
    return lines


def triangle_to_square(pt):
    """
    Args:
        pt (tuple) - (left, top, right) triangle point
                e.g., (1, 0, 0) is left corner
                e.g., (0, 1, 0) is top corner

    Returns:
        pt (tuple) - (x, y) same point in 2D space
    """
    converter = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    new_pt = np.dot(np.array(pt[:2]), converter)
    return new_pt.transpose()


def cmpd_to_pt_canvp(cmpd, square=False):
    """
    Tentative function only works for CaNaVP system

    """
    ca = CompAnalyzer(cmpd)
    reduced_ratio = 72 / (ca.amt_of_el('O'))
    Ca_ratio = ca.amt_of_el('Ca') * reduced_ratio / 9
    Na_ratio = ca.amt_of_el('Na') * reduced_ratio / 18
    pt = [Ca_ratio, Na_ratio, 1 - Ca_ratio - Na_ratio]
    if square == False:
        return pt
    else:
        return triangle_to_square(tuple(pt))


def convert_comp_to_content(cmpd, square=False):
    """
    Tentative function only works for CaNaVP system

    """
    ca = CompAnalyzer(cmpd)
    reduced_ratio = (ca.amt_of_el('O')) / 12
    Ca = ca.amt_of_el('Ca') / reduced_ratio
    Na = ca.amt_of_el('Na') / reduced_ratio
    pt = np.array([Ca, Na, 17])
    tot = np.sum(pt)

    return pt, tot


def unique_lines(q):
    """
    Given all the facets, convert it into a set of unique lines.  Specifically
    used for converting convex hull facets into line pairs of coordinates.

    Args:
        q: A 2-dim sequence, where each row represents a facet. E.g.,
            [[1,2,3],[3,6,7],...]

    Returns:
        setoflines:
            A set of tuple of lines.  E.g., ((1,2), (1,3), (2,3), ....)
    """
    setoflines = set()
    for facets in q:
        for line in itertools.combinations(facets, 2):
            setoflines.add(tuple(sorted(line)))
    return setoflines


def params():
    """
    Args:

    Returns:
        just colors for stable and unstable points
    """
    return {'stable': {'c': tableau_colors()['blue'],
                       'm': 'o',
                       'alpha': 1},
            'unstable': {'c': tableau_colors()['red'],
                         'm': '^',
                         'alpha': 0.0}}


def set_rc_params():
    """
    General params for plot.
    """
    params = {'axes.linewidth': 1.5,
              'axes.unicode_minus': False,
              'figure.dpi': 300,
              'font.size': 25,
              'legend.frameon': False,
              'legend.handletextpad': 0.4,
              'legend.handlelength': 1,
              'legend.fontsize': 10,
              'lines.markeredgewidth': 4,
              'lines.linewidth': 3,
              'lines.markersize': 15,
              'mathtext.default': 'regular',
              'savefig.bbox': 'tight',
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'xtick.major.size': 5,
              'xtick.minor.size': 3,
              'ytick.major.size': 5,
              'ytick.minor.size': 3,
              'xtick.major.width': 1,
              'xtick.minor.width': 0.5,
              'ytick.major.width': 1,
              'ytick.minor.width': 0.5,
              'xtick.top': True,
              'ytick.right': True,
              'axes.edgecolor': 'black',
              'legend.fancybox': True,
              'figure.figsize': [8, 6]}
    for p in params:
        mpl.rcParams[p] = params[p]


def get_reduced_traj(to_triangle=True):

    traj_data_path = '/Users/yun/Desktop/github_codes/CaNaVP/data/traj_reformatted.json'
    traj_data = read_json(traj_data_path)
    reduced_traj_data = {}

    for traj in traj_data:
        reduced_traj_data[traj] = [(0.5, 1.0)]
        for i, number in enumerate(traj_data[traj]['Na']):
            # if number == traj_data[traj]['Na_last'] and traj_data[traj]['Ca'][i] == traj_data[traj]['Ca_last']:
            #     break
            # else:
            reduced_traj_data[traj].append((traj_data[traj]['Ca'][i] / 120, number / 120))

    if to_triangle:
        reduced_traj_in_tri = {}
        for i in reduced_traj_data:
            reduced_traj_in_tri[i] = []
            for j in reduced_traj_data[i]:
                pt = [j[0] / 1.5, j[1] / 3, 1 - j[0] / 1.5 - j[0] / 3]
                pt = triangle_to_square(pt)
                reduced_traj_in_tri[i].append(pt)
        return reduced_traj_in_tri

    return reduced_traj_data


def quick_check():

    traj_data_path = '/Users/yun/Desktop/github_codes/CaNaVP/data/0728_preliminary_ce/' \
                     'traj_data.json'
    traj_data = read_json(traj_data_path)

    print(len(traj_data))

    return


if __name__ == '__main__':

    quick_check()