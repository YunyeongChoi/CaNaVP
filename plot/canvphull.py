#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 2022

@author: yun
@purpose: Plotting CaNaVP ternary phase diagram.
"""

import json
from compmatscipy.handy_functions import H_from_E
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from compmatscipy.HullAnalysis import AnalyzeHull, GetHullInputData
import matplotlib.tri as mtri

from plot.chempo_utils import ternary_chempo
from plot.plot_utils import *

FIG_DIR = os.getcwd()
DATA_DIR = FIG_DIR.replace('plot', 'data')
vasp_data = read_json(os.path.join(DATA_DIR, 'final_data', 'dft', '031623_ce_data.json'))

# For further use. Currently not MP compatible.
# MP_DIR = os.path.join(DATA_DIR, '0414_MP.json')

# Correction values are based on MP2020Compatibility.yaml (+U setting)
# https://github.com/materialsproject/pymatgen/blob/master/pymatgen/entries/MP2020Compatibility.yaml
# Chemical potential is from stable compounds in MP. (eV/atom)
V_corr, O_corr = -1.7, -0.687
mus = {'Ca': -1.9985,
       'Na': -1.3122,
       'V': -9.0824 - V_corr,
       'P': -5.4133,
       'O': -4.9480 - O_corr}


def polish_data() -> dict:
    """
    Polishing data to the form of {'Structure formula': [energy_list]}.
    """
    polished_dict = {}

    for data in vasp_data:
        if vasp_data[data]['convergence'] and len(vasp_data[data]['errors']) == 0:
            contcar = Structure.from_dict(json.loads(vasp_data[data]['contcar']))
            key = contcar.formula
            if key not in polished_dict:
                polished_dict[key] = []
            polished_dict[key].append(vasp_data[data]['energy'])

    return polished_dict


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


def ternary_pd(hull_data, line_data, exp=True, traj=True, traj_avg=True):

    fig = plt.figure(dpi=300)

    triangle_boundary()

    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.1, 1.0])
    for spine in ['bottom', 'left', 'top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().tick_params(bottom=False, top=False, left=False, right=False)

    # Experimental data
    if exp:
        exp_stable_list = [(1, 1), (0, 3), (0, 1), (1, 0), (1.5, 0), (0.5, 2.0), (0.4, 0.6)]
        exp_unstable_list = [(0.93, 0.65), (0.5, 1), (0.6, 1), (0.7, 0.8), (0.7, 0.2)]
        for config in exp_stable_list:
            pt = [config[0] / 1.5, config[1] / 3, 1 - config[0] / 1.5 - config[0] / 3]
            pt = triangle_to_square(pt)
            plt.scatter(pt[0], pt[1], marker='*', s=64, color='blue', zorder=5,
                             edgecolors='black', linewidths=1)
        for config in exp_unstable_list:
            pt = [config[0] / 1.5, config[1] / 3, 1 - config[0] / 1.5 - config[0] / 3]
            pt = triangle_to_square(pt)
            plt.scatter(pt[0], pt[1], marker='*', s=64, color='red', zorder=5,
                             edgecolors='black', linewidths=1)

    # Plotting gcMC trajectories.
    if traj:
        traj_dict = new_get_traj()
        plt.scatter(0.5, 1.0, s=20, color='black', zorder=6)
        for trajs in traj_dict:
            for i, pt in enumerate(traj_dict[trajs]):
                if not i == len(traj_dict[trajs]) - 1:
                    plt.plot((traj_dict[trajs][i][0], traj_dict[trajs][i+1][0]),
                             (traj_dict[trajs][i][1], traj_dict[trajs][i+1][1]),
                             '--', linewidth=1, zorder=4, color='darkgreen')
            plt.scatter(traj_dict[trajs][-1][0], traj_dict[trajs][-1][1], s=30, color='red',
                        zorder=6)

    if traj_avg:
        traj_dict = from_energy_concen()
        for chempo in traj_dict:
            plt.scatter(traj_dict[chempo][0], traj_dict[chempo][1], s=30, color='red',
                        edgecolors='black', zorder=5)
            Ca_chempo = round(float(chempo.split('_')[0]), 2)
            Na_chempo = round(float(chempo.split('_')[1]), 2)
            chempo_str = str(Ca_chempo) + ', ' + str(Na_chempo)
            plt.annotate(chempo_str, (traj_dict[chempo][0], traj_dict[chempo][1]), size=15)

    # Plotting ground states.
    cmpd_params = params()
    for cmpd in hull_data:
        if len(CompAnalyzer(cmpd).els) == 1:
            continue
        pt = cmpd_to_pt_canvp(cmpd)
        pt = triangle_to_square(pt)
        stability = 'stable' if hull_data[cmpd]['stability'] else 'unstable'
        color, marker, alpha = [cmpd_params[stability][k] for k in ['c', 'm', 'alpha']]

        plt.scatter(pt[0], pt[1],
                    color='white',
                    marker=marker,
                    facecolor='white',
                    edgecolor=color,
                    alpha=alpha,
                    s=64,
                    zorder=4)


    # Plotting tie lines.
    for l in line_data:
        xy = [cmpd_to_pt_canvp(line_data[l]['cmpds'][0], True),
              cmpd_to_pt_canvp(line_data[l]['cmpds'][1], True)]
        # line_data[l]['pts'] = tuple([cmpd_to_pt(cmpd, els) for cmpd in cmpds])
        x = (xy[0][0], xy[1][0])
        y = (xy[0][1], xy[1][1])
        print(x, y)
        plt.plot(x, y, zorder=2, lw=1.5, color='black')

    # Plotting energy surface.
    x, y, z = [], [], []

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

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    T = mtri.Triangulation(x, y)
    normi = mpl.colors.Normalize(vmin=0, vmax=350)
    plt.tricontourf(x, y, T.triangles, z, levels=20, cmap='plasma_r', alpha=1, zorder=1, norm=normi)

    # Plotting labels and lagend.
    cmap = 'plasma_r'
    vmin = 0
    vmax = 350
    # vmax = max(z)
    label = r'$\Delta$' + r'$\it{E}_{d}$' + r'$\/(\frac{meV}{V_{2}(PO_{4})_{3}})$'
    ticks = np.arange(0, vmax, 100)
    position = (0.85, 0.22, 0.04, 0.55)
    label_size = 10
    tick_size = 13
    tick_len = 2.5
    tick_width = 1

    cmap = mpl.cm.get_cmap(cmap)
    # norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes(position)
    cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=normi, orientation='vertical', alpha=1)
    cb.set_label(label, fontsize=label_size)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(ticks)
    cb.ax.tick_params(labelsize=tick_size, length=tick_len, width=tick_width)

    # plt.show()
    fig.savefig("/Users/yun/Berkeley/Codes/CaNaVP/test/gcmc_0320_300K.png")

    return


def test_get_triangles():

    stable_list = []
    triangles = {}
    hull_data, line_data = get_hull_data(convert_data(vasp_data))
    for i in hull_data:
        if hull_data[i]['stability'] and i not in ['Ca', 'O', 'Na', 'P', 'V']:
            stable_list.append(i)

    return stable_list


if __name__ == '__main__':
    set_rc_params()
    vasp_data = convert_data(vasp_data)
    hull_data, line_data = get_hull_data(vasp_data)

    polished_line_data = []
    for i in line_data:
        polished_line_data.append(line_data[i]['cmpds'])
    tc = ternary_chempo(polished_line_data, vasp_data)
    chempo_data = tc.get_chempo_at_cycles()
    # print(chempo_data)
    ternary_pd(hull_data, line_data, traj=False)

