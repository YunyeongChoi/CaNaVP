#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 2022

@author: yun
@purpose: Plotting CaNaVP ternary phase diagram from eci and wragler.
"""

import os
import pickle
import itertools
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from smol.cofe.wrangling.wrangler import StructureWrangler
from compmatscipy.handy_functions import H_from_E, read_json
from compmatscipy.plotting_functions import tableau_colors
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.HullAnalysis import AnalyzeHull, GetHullInputData

DATA_DIR = ''
eci_path = '/Users/yun/Desktop/github_codes/CaNaVP/data/0728_preliminary_ce/ecis_gs'
wrangler_path = '/Users/yun/Desktop/github_codes/CaNaVP/data/0728_preliminary_ce/prelim_canvp_ce.json'

# Correction values are based on MP2020Compatibility.yaml (+U setting)
# https://github.com/materialsproject/pymatgen/blob/master/pymatgen/entries/MP2020Compatibility.yaml
# Chemical potential is from stable compounds in MP. (eV/atom)
V_corr, O_corr = -1.7, -0.687
mus = {'Ca': -1.9985,
       'Na': -1.3122,
       'V': -9.0824 - V_corr,
       'P': -5.4133,
       'O': -4.9480 - O_corr}


def get_fitted_energies(ecifile, wranglerfile) -> dict:
    """
    Args:
        ecifile: ECI filename in string format. Load ECI using pickle.
        wranglerfile: Subspace filename in string format.
    Returns:
        Energy dictionary, {"Structure.formula": list[float]}
        energy unit is eV / atom.
    """
    eci = pickle.load(open(ecifile, 'rb'))
    wrangler = StructureWrangler.from_dict(read_json(wranglerfile))

    fitted_energy = {}
    for entry in wrangler.entries:
        try:
            ce_energy = eci @ entry.data['correlations'] * entry.data['size'] / \
                        entry.structure.num_sites
        except ValueError:
            print("{} cannot be matched. Check wrangler data".format(entry.structure.formula))
            continue

        if entry.structure.formula not in fitted_energy:
            fitted_energy[entry.structure.formula] = [ce_energy]
        else:
            fitted_energy[entry.structure.formula].append(ce_energy)

    return fitted_energy


def convert_data(d) -> dict:
    """
    Args:
        d: dictionary with form {"Structure.formula": list[float]}
    Returns:
        dictionary with form {CompAnalyzer.std_formula : energy per atom}
    """
    converted_data = {}
    for keys in d:
        converted_data[CompAnalyzer(keys).std_formula()] = min(d[keys])

    return converted_data


def get_hull_data(d, remake=True) -> (dict, dict):
    """
    Args:
        d: dictionary with form {CompAnalyzer.std_formula : energy per atom}
        remake: remake data or get from saved data.
    Returns:
        tuple(dict, dict). hull data and line data.
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
        None.
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
    Tentative function only works for CaNaVP system.
    """
    ca = CompAnalyzer(cmpd)
    reduced_ratio = 72 / (ca.amt_of_el('O'))
    ca_ratio = ca.amt_of_el('Ca') * reduced_ratio / 9
    na_ratio = ca.amt_of_el('Na') * reduced_ratio / 18
    pt = [ca_ratio, na_ratio, 1 - ca_ratio - na_ratio]
    if not square:
        return pt
    else:
        return triangle_to_square(tuple(pt))


def convert_comp_to_content(cmpd):
    """
    Tentative function only works for CaNaVP system.
    """
    ca = CompAnalyzer(cmpd)
    reduced_ratio = (ca.amt_of_el('O')) / 12
    ca_number = ca.amt_of_el('Ca') / reduced_ratio
    na_number = ca.amt_of_el('Na') / reduced_ratio
    pt = np.array([ca_number, na_number, 17])
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


def ternary_pd(hull_data, line_data, save_path, option='decompose'):
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
    ax = plt.tricontourf(x, y, T.triangles, z, levels=20, cmap='plasma_r', alpha=1, zorder=1)

    cmap = 'plasma_r'
    vmin = 0
    vmax = max(z)
    label = r'$\Delta$' + r'$\it{E}_{d}$' + r'$\/(\frac{meV}{V_{2}(PO_{4})_{3}})$'
    ticks = np.arange(0, vmax, 100)
    position = (0.85, 0.22, 0.04, 0.55)
    label_size = 10
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

    # ax = plt.show()
    fig.savefig(save_path)

    return ax, x, y, z


# Functions for plot setting

def params():
    """
    Args:
        None.
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


def main(eci_path, wrangler_path, save_path):
    set_rc_params()
    ce_fitted_energy = get_fitted_energies(eci_path, wrangler_path)
    hull, line = get_hull_data(convert_data(ce_fitted_energy))
    ternary_pd(hull, line, save_path)


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=str, required=False, default='',
                        help="ECI file path.")
    parser.add_argument('-w', type=str, required=False, default='',
                        help="Wrangler file path.")
    parser.add_argument('-o', type=str, required=False, default='fast',
                        help="Plot file path to save.")
    args = parser.parse_args()
    main(args.e, args.w, args.o)
    """
    main(eci_path, wrangler_path, './cetest_png')
