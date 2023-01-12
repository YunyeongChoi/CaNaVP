#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 2022

@author: yun
@purpose: Plotting CaNaVP ternary phase diagram from eci and wragler.
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from smol.cofe.wrangling.wrangler import StructureWrangler
from compmatscipy.handy_functions import H_from_E, read_json
from compmatscipy.HullAnalysis import AnalyzeHull, GetHullInputData
from plot.plot_utils import *

DATA_DIR = ''
eci_path = '/Users/yun/Desktop/github_codes/CaNaVP/data/final_data/ce/1003_eci_gp'
wrangler_path = '/Users/yun/Desktop/github_codes/CaNaVP/data/final_data/ce/0927_wrangler.json'

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
    # cmpds = sorted(list(form_energy.keys()))

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


def ternary_pd(hull_data, line_data, save_path, exp=True, traj=True):

    fig = plt.figure(dpi=300)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.1, 1.0])
    for spine in ['bottom', 'left', 'top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().tick_params(bottom=False, top=False, left=False, right=False)

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
                    zorder=3)

    if exp:  # Experimental data
        exp_stable_list = [(1, 1), (0, 3), (0, 1), (1, 0), (1.5, 0), (0.5, 2.0), (0.4, 0.6)]
        exp_unstable_list = [(0.93, 0.65), (0.5, 1), (0.6, 1), (0.7, 0.8), (0.7, 0.2)]
        for config in exp_stable_list:
            pt = [config[0] / 1.5, config[1] / 3, 1 - config[0] / 1.5 - config[0] / 3]
            pt = triangle_to_square(pt)
            plt.scatter(pt[0], pt[1], marker='*', s=64, color='blue', zorder=3,
                        edgecolors='black', linewidths=1)
        for config in exp_unstable_list:
            pt = [config[0] / 1.5, config[1] / 3, 1 - config[0] / 1.5 - config[0] / 3]
            pt = triangle_to_square(pt)
            plt.scatter(pt[0], pt[1], marker='*', s=64, color='red', zorder=3,
                        edgecolors='black', linewidths=1)

    if traj:
        traj_dict = new_get_traj()
        startpt = [0.5 / 1.5, 1.0 / 3, 1 - 0.5 / 1.5 - 1.0 / 3]
        # startpt = [66 / 120 / 1.5, 111 / 120 / 3, 1 - 66 / 120 / 1.5 - 111 / 120 / 3]
        startpt = triangle_to_square(startpt)
        # plt.scatter(startpt[0], startpt[1], s=36, color='black', zorder=6,
        #             edgecolors='black', linewidths=1)
        for trajs in traj_dict:
            for i, pt in enumerate(traj_dict[trajs]):
                if not i == len(traj_dict[trajs]) - 1:
                    plt.plot((traj_dict[trajs][i][0], traj_dict[trajs][i+1][0]),
                             (traj_dict[trajs][i][1], traj_dict[trajs][i+1][1]),
                             '--', linewidth=1, zorder=4, color='darkslategrey')
            plt.scatter(traj_dict[trajs][-1][0], traj_dict[trajs][-1][1], s=36, color='red',
                        zorder=5, edgecolors='black', linewidths=1)
            # plt.text(pt[0] + 0.02, pt[1] + 0.02, str(trajs), fontsize=5, zorder=5)

    for l in line_data:
        xy = [cmpd_to_pt_canvp(line_data[l]['cmpds'][0], True),
              cmpd_to_pt_canvp(line_data[l]['cmpds'][1], True)]
        # line_data[l]['pts'] = tuple([cmpd_to_pt(cmpd, els) for cmpd in cmpds])
        x = (xy[0][0], xy[1][0])
        y = (xy[0][1], xy[1][1])
        plt.plot(x, y, zorder=2, lw=1.5, color='black')

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
    ax = plt.tricontourf(x, y, T.triangles, z, levels=20, cmap='plasma_r',
                         alpha=1, zorder=1, norm=normi)

    cmap = 'plasma_r'
    vmin = 0
    vmax = 350
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
    fig.savefig(save_path)

    return ax, x, y, z


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
    main(eci_path, wrangler_path, '../data/300K_voltage_traj')
