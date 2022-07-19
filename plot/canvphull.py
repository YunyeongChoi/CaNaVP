#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 2022

@author: yun
@purpose: Plotting CaNaVP ternary phase diagram.
"""

import os
import json
import numpy as np
from src.setter import read_json, write_json
from compmatscipy.handy_functions import H_from_E
import matplotlib as mpl
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from compmatscipy.plotting_functions import tableau_colors
from pymatgen.core.composition import Composition
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.HullAnalysis import AnalyzeHull, GetHullInputData
from compmatscipy.PullMP import PullMP
import itertools
import matplotlib.tri as mtri

FIG_DIR = os.getcwd()
DATA_DIR = FIG_DIR.replace('plot', 'data')
vasp_data = read_json(os.path.join(DATA_DIR, '0718_preliminary.json'))
# For further use. Currently not MP compatible.
# MP_DIR = os.path.join(DATA_DIR, '0414_MP.json')

# Correction values are based on MP2020Compatibility.yaml (+U setting)
# https://github.com/materialsproject/pymatgen/blob/master/pymatgen/entries/MP2020Compatibility.yaml
# Chemical potential is from stable compounds in MP. (eV/atom)
V_corr, O_corr = -1.7, -0.687
mus =  {'Ca': -1.9985,
        'Na': -1.3122,
        'V': -9.0824 - V_corr,
        'P': -5.4133,
        'O': -4.9480 - O_corr}


def polish_data() -> dict:
    """
    Polishing data to the form of {'Structure formula': [energy_list]}.
    """
    polished_dict = {}

    for i in vasp_data:
        if vasp_data[i]['convergence'] and len(vasp_data[i]['errors']) == 0:
            contcar = Structure.from_dict(json.loads(vasp_data[i]['contcar']))
            key = contcar.formula
            if not key in polished_dict:
                polished_dict[key] = []
            polished_dict[key].append(vasp_data[i]['energy'])

    return polished_dict


def convert_data(d):
    """
    Convert polished raw data to {std_formula (based on CompAnalyzer) : energy per atom}
    """
    converted_data = {}
    for keys in d:
        converted_data[CompAnalyzer(keys).std_formula()] = min(d[keys])

    return converted_data


def get_hull_data(d, remake=True) -> (dict, dict):
    """
    :param d: converted data want to generate hull.
    :param remake: remake data or get from saved data.
    :return: First is hull data, second is hull line data.
    """
    fjson_hull = os.path.join(DATA_DIR, 'hull.json')
    fjson_line = os.path.join(DATA_DIR, 'line.json')
    if not remake:
        return read_json(fjson_hull), read_json(fjson_line)

    line_data = {}
    form_energy = {key: {'Ef': H_from_E(CompAnalyzer(key).els_to_amts(), d[key], mus)} for key in d}
    ghid = GetHullInputData(form_energy, 'Ef')
    hullin = ghid.hull_data(remake=True)
    cmpds = sorted(list(form_energy.keys()))

    for space in hullin:
        ah = AnalyzeHull(hullin, space)
        hull_data = ah.hull_output_data
        simplices = ah.hull_simplices
        lines = unique_lines(simplices)
        sorted_cmpds = ah.sorted_compounds
        stable_cmpds = ah.stable_compounds
        for l in lines:
            cmpds = (sorted_cmpds[l[0]], sorted_cmpds[l[1]])
            if (cmpds[0] not in stable_cmpds) or (cmpds[1] not in stable_cmpds):
                continue
            if cmpds[0] in ['Ca', 'Na', 'V', 'O', 'P'] or cmpds[1] in ['Ca', 'Na', 'V', 'O', 'P']:
                continue
            # print(cmpds)
            key = '_'.join([str(x) for x in l])
            line_data[key] = {'cmpds': cmpds}

    # write_json(hull_data, fjson_hull)
    # write_json(line_data, fjson_line)

    return hull_data, line_data


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


def ternary_pd(hull_data, line_data, option='decompose'):

    fig = plt.figure(dpi=300)

    lines = triangle_boundary()

    ax = plt.xlim([-0.2, 1.2])
    ax = plt.ylim([-0.1, 1.0])
    for spine in ['bottom', 'left', 'top', 'right']:
        ax = plt.gca().spines[spine].set_visible(False)
    ax = plt.gca().xaxis.set_ticklabels([])
    ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.gca().tick_params(bottom=False, top=False, left=False, right=False)

    cmpd_params = params()
    for cmpd in hull_data:
        if len(CompAnalyzer(cmpd).els) == 1:
            continue
        pt = cmpd_to_pt_canvp(cmpd)
        pt = triangle_to_square(pt)
        stability = 'stable' if hull_data[cmpd]['stability'] else 'unstable'
        color, marker, alpha = [cmpd_params[stability][k] for k in ['c', 'm', 'alpha']]
        ax = plt.scatter(pt[0], pt[1],
                         color='white',
                         marker=marker,
                         facecolor='white',
                         edgecolor=color,
                         alpha=alpha,
                         s=64,
                         zorder=2)

    if True:  # Experimental data
        exp_stable_list = [(1, 1), (0, 3), (0, 1), (1, 0), (1.5, 0), (0.5, 2.0), (0.4, 0.6)]
        exp_unstable_list = [(0.93, 0.65), (0.5, 1), (0.6, 1), (0.7, 0.8), (0.7, 0.2)]
        for config in exp_stable_list:
            pt = [config[0] / 1.5, config[1] / 3, 1 - config[0] / 1.5 - config[0] / 3]
            pt = triangle_to_square(pt)
            ax = plt.scatter(pt[0], pt[1], marker='*', s=64, color='blue', zorder=3,
                             edgecolors='black', linewidths=1)
        for config in exp_unstable_list:
            pt = [config[0] / 1.5, config[1] / 3, 1 - config[0] / 1.5 - config[0] / 3]
            pt = triangle_to_square(pt)
            ax = plt.scatter(pt[0], pt[1], marker='*', s=64, color='red', zorder=3,
                             edgecolors='black', linewidths=1)

    for l in line_data:
        xy = [cmpd_to_pt_canvp(line_data[l]['cmpds'][0], True),
              cmpd_to_pt_canvp(line_data[l]['cmpds'][1], True)]
        print(xy)
        # line_data[l]['pts'] = tuple([cmpd_to_pt(cmpd, els) for cmpd in cmpds])
        x = (xy[0][0], xy[1][0])
        y = (xy[0][1], xy[1][1])
        print(x, y)
        ax = plt.plot(x, y, zorder=1, lw=1.5, color='black')

    x = []
    y = []
    z = []

    for key in hull_data:
        if key not in ['Ca', 'Na', 'V', 'O', 'P']:
            xy = triangle_to_square(cmpd_to_pt_canvp(key))
            atom_num, atom_tot = convert_comp_to_content(key)
            x.append(xy[0])
            y.append(xy[1])
            if hull_data[key]['Ed'] < 0:
                z.append(0)
            else:
                z.append(hull_data[key]['Ed'] * 1000 * atom_tot)

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    T = mtri.Triangulation(x, y)
    ax = plt.tricontourf(x, y, T.triangles, z, levels=15, cmap='plasma_r', alpha=1, zorder=1)

    cmap = 'plasma_r'
    vmin = 0
    vmax = 1000
    label = r'$\Delta$' + r'$\it{E}_{d}$' + r'$\/(\frac{meV}{V_{2}(PO_{4})_{3}})$'
    ticks = (0, 200, 400, 600, 800, 1000)
    position = (0.85, 0.22, 0.04, 0.55)
    label_size = 15
    tick_size = 13
    tick_len = 2.5
    tick_width = 1

    cmap = mpl.cm.get_cmap(cmap)
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes(position)
    cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation='vertical', alpha=1)
    cb.set_label(label, fontsize=label_size)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(ticks)
    cb.ax.tick_params(labelsize=tick_size, length=tick_len, width=tick_width)
    ax = plt.show()

    return ax, x, y, z


if __name__ == '__main__':
    set_rc_params()
    a, b = get_hull_data(convert_data(polish_data()))
    ternary_pd(a, b)