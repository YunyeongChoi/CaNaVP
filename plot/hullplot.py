#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:13:20 2021

@author: yun
"""

import os
import numpy as np
from glob import glob
from copy import deepcopy
from compmatscipy.handy_functions import read_json, write_json, H_from_E
from compmatscipy.HelpWithVASP import VASPSetUp, VASPBasicAnalysis, JobSubmission, magnetic_els
import matplotlib.pyplot as plt
import matplotlib as mpl
from compmatscipy.plotting_functions import set_rc_params, tableau_colors
from pymatgen.core.composition import Composition
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.HullAnalysis import AnalyzeHull, GetHullInputData
from compmatscipy.PullMP import PullMP
import itertools
import matplotlib.tri as mtri



MAIN_DIR = os.getcwd()
DATA_DIR = MAIN_DIR.replace('scripts', 'data')
FIG_DIR = MAIN_DIR.replace('scripts', 'figure')
MP_DIR = os.path.join(DATA_DIR, '0414_MP.json')

V_corr, O_corr = -1.682, -0.70229 # +U for these energies
mus_old =  {'Ca' : -2.0056,
            'Na': -1.3225,
            'V' : -9.0839 - V_corr,
            'P' : -5.4133,
            'O' : -4.9480 - O_corr}

def convert_data(d):
    '''
    Convert raw data to {std_forule : energy per atom}
    '''
    converted_data = {}
    for keys in d:
        converted_data[CompAnalyzer(keys).std_formula()] = min(d[keys])
    
    return converted_data

def query_MP(fjson = MP_DIR, remake=False):
    '''
    Query MP Ca-Na-V-P-O space
    Convert key to {std_formula : energy per atom}
    '''
    if remake:
        MP = {}
        reform = {}
        a = PullMP('R0CmAuNPKWzrUo8Z')
        query = a.specific_hull_query(fjson=fjson, elements = ['Ca','Na','V','P','O'], props = ["energy_per_atom", "pretty_formula", "formation_energy_per_atom", ], remake=remake, write_it=False, remove_polymorphs=True)
        for keys in query:
            reform[query[keys]['pretty_formula']] = query[keys]['energy_per_atom']
        for keys in reform:
            MP[CompAnalyzer(keys).std_formula()] = reform[keys]
        write_json(MP, fjson)
        return reform
    
    else:
        MP = read_json(fjson)
        return MP
    
def get_total():
    '''
    Merge MP and Calculated data
    {std_formula : energy per atom}
    '''
    d = read_json(os.path.join(DATA_DIR, '0201_0K_DFT_energy.json'))
    converted_data = convert_data(d) # Raw data to unified format
    MP = query_MP(remake=False) # Query from MP

    total = {}
    for key in converted_data:
        total[key] = converted_data[key]
    for key in MP:
        if key in total.keys():
            if MP[key] < total[key]:
                total[key] = MP[key]
        else:
            total[key] = MP[key]
    
    #total['Ca7O48P12V8'] = -7.533
    total['Ca2O12P3V2'] = -7.633
    return total

def get_triangle():
    '''
    Get energy per atom for compounds in the triangle plot
    '''
    
    inside = {}
    target_list = []
    total = get_total()
    
    for key in total:
        els_dict = CompAnalyzer(key).els_to_amts()
        if 'V' in els_dict and 'P' in els_dict and 'O' in els_dict:
            if els_dict['O']/els_dict['V'] != 6:
                continue
            elif els_dict['O']/els_dict['P'] != 4:
                continue
            elif els_dict['P']/els_dict['V'] != 1.5:
                continue
            elif 'Ca' in els_dict and els_dict['Ca']/els_dict['V'] > 1.5/2:
                continue
            elif 'Na' in els_dict and els_dict['Na']/els_dict['V'] > 3/2:
                continue
            else:
                target_list.append(key)
        else:
            continue
    
    for keys in target_list:
        inside[keys] = total[keys]

    return inside
    
def get_hullandline(d, option='inside', remake=True):
    
    if option == 'inside':
        fjson_hull=os.path.join(DATA_DIR, '0201_0K_Ehull_inside.json')
        fjson_line=os.path.join(DATA_DIR, '0201_0K_line_inside.json')
    else:
        fjson_hull=os.path.join(DATA_DIR, '0201_0K_Ehull_total.json')
        fjson_line=os.path.join(DATA_DIR, '0201_0K_line_total.json')
    if not remake:
        return read_json(fjson_hull), read_json(fjson_line)
    
    line_data = {}

    new = {key : {'Ef' : H_from_E(CompAnalyzer(key).els_to_amts(), 
                                             d[key], 
                                             mus_old)} for key in d}
    ghid = GetHullInputData(new, 'Ef')
    hullin = ghid.hull_data(remake=True)
    cmpds = sorted(list(new.keys()))
    for space in hullin:
        ah = AnalyzeHull(hullin, space)
        hull_data = ah.hull_output_data
        simplices = ah.hull_simplices
        lines = uniquelines(simplices)
        sorted_cmpds = ah.sorted_compounds
        stable_cmpds = ah.stable_compounds
        for l in lines:
            cmpds = (sorted_cmpds[l[0]], sorted_cmpds[l[1]])
            if (cmpds[0] not in stable_cmpds) or (cmpds[1] not in stable_cmpds):
                continue
            if cmpds[0] in ['Ca', 'Na', 'V', 'O', 'P'] or cmpds[1] in ['Ca', 'Na', 'V', 'O', 'P']:
                continue
            #print(cmpds)
            key = '_'.join([str(x) for x in l])
            line_data[key] = {'cmpds' : cmpds}

    write_json(hull_data, fjson_hull)
    write_json(line_data, fjson_line)
    
    return hull_data, line_data

def trianglePD(hull_data, line_data, els = ['Ca3O24P6V4', 'Na3O12P3V2', 'O12P3V2'], tri_lw=1.5, option='decompose'):
    
    fig = plt.figure(dpi=200)
    inside = get_triangle()

    lines = triangle_boundary()
    for l in lines:
        continue
        ax = plt.plot(lines[l]['x'], lines[l]['y'],
                      color='black',
                      lw=tri_lw,
                      ls='--',
                      zorder=1)
    ax = plt.xlim([-0.2,1.2])
    ax = plt.ylim([-0.1,1.0])
    for spine in ['bottom', 'left', 'top', 'right']:
        ax = plt.gca().spines[spine].set_visible(False)
    ax = plt.gca().xaxis.set_ticklabels([])
    ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.gca().tick_params(bottom=False, top=False, left=False, right=False)
    
    cmpd_params = params()
    for cmpd in hull_data:
        if len(CompAnalyzer(cmpd).els) == 1:
            continue
        if cmpd in inside:
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
                             s =100,
                             zorder=2)    
    
    if True: # Experimental data
        exp_stable_list = [(1, 1), (0, 3), (0, 1), (1, 0), (1.5, 0), (0.5, 2.0), (0.4, 0.6)]
        exp_unstable_list = [(0.93, 0.65), (0.5, 1), (0.6, 1), (0.7, 0.8), (0.7, 0.2)]
        for config in exp_stable_list:
            pt = [config[0]/1.5, config[1]/3, 1-config[0]/1.5-config[0]/3]
            pt = triangle_to_square(pt)
            ax = plt.scatter(pt[0], pt[1], marker='*', s=100, color='blue', zorder=3)
        for config in exp_unstable_list:
            pt = [config[0]/1.5, config[1]/3, 1-config[0]/1.5-config[0]/3]
            pt = triangle_to_square(pt)
            ax = plt.scatter(pt[0], pt[1], marker='*', s=100, color='red', zorder=3)

    for l in line_data:
        if line_data[l]['cmpds'][0] in inside and line_data[l]['cmpds'][1] in inside:
            xy = [cmpd_to_pt_canvp(line_data[l]['cmpds'][0], True), cmpd_to_pt_canvp(line_data[l]['cmpds'][1], True)]
        else:
            continue
        print(xy)
        #line_data[l]['pts'] = tuple([cmpd_to_pt(cmpd, els) for cmpd in cmpds])
        x = (xy[0][0], xy[1][0])
        y = (xy[0][1], xy[1][1])
        print(x,y)
        ax = plt.plot(x, y, zorder=1, lw=1.5, color='black')
        
    x = []
    y = []
    z = []
    
    if option == 'decompose':
        
        for key in hull_data:
            if key not in ['Ca', 'Na', 'V', 'O', 'P'] and key in inside:
                xy = triangle_to_square(cmpd_to_pt_canvp(key))
                atom_num, atom_tot = convert_comp_to_content(key)
                x.append(xy[0])
                y.append(xy[1])
                if hull_data[key]['Ed'] < 0:
                    z.append(0)
                else:
                    z.append(hull_data[key]['Ed']*1000*atom_tot)
    
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        T =  mtri.Triangulation(x, y)
        ax = plt.tricontourf(x,y,T.triangles, z, levels=15, cmap='plasma_r',alpha=1,zorder=1)
        
        cmap='plasma_r'
        vmin = 0
        vmax = 1000
        label = r'$\Delta$'+r'$\it{E}_{d}$'+r'$\/(\frac{meV}{V_{2}(PO_{4})_{3}})$'
        ticks = (0, 200, 400, 600, 800, 1000)
        
    else:
        Ca_E = inside['Ca3O24P6V4']
        Na_E = inside['Na3O12P3V2']
        Zero_E = inside['O12P3V2']
        form_E_dict = {}
        for keys in inside:
            composition = cmpd_to_pt_canvp(keys)
            form_E = inside[keys] - composition[0] * Ca_E - composition[1] * Na_E - composition[2] * Zero_E
            form_E_dict[keys] = form_E*1000

        for key in hull_data:
            if key not in ['Ca', 'Na', 'V', 'O', 'P']:
                xy = triangle_to_square(cmpd_to_pt_canvp(key))
                x.append(xy[0])
                y.append(xy[1])
                z.append(form_E_dict[key])
        
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        T =  mtri.Triangulation(x, y)
        ax = plt.tricontourf(x,y,T.triangles, z, levels=15, cmap='plasma_r',alpha=1,zorder=1)
        
        cmap='plasma_r'
        vmin = -40
        vmax = 40
        label = r'$\Delta$'+r'$\it{E}_{f}$'+r'$\/(\frac{meV}{atom})$'
        ticks = (-40, -20, 0, 20, 40)
        
    position = (0.85, 0.22, 0.04, 0.55)
    label_size=15
    tick_size=13
    tick_len=2.5
    tick_width=1
    
    cmap = mpl.cm.get_cmap(cmap)
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes(position)    
    cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation='vertical', alpha=1)
    cb.set_label(label, fontsize=label_size)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(ticks)
    cb.ax.tick_params(labelsize=tick_size, length=tick_len, width=tick_width)
    ax = plt.show()
    
    return ax,x,y,z
    

def uniquelines(q):
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

def triangle_boundary():
    """
    Args:
        
    Returns:
        the boundary of a triangle where each axis goes 0 to 1
    """
    corners = [(0,0,0), (0,1,0), (1,0,0)]
    corners = [triangle_to_square(pt) for pt in corners]
    data = dict(zip(['left', 'top', 'right'], corners))
    lines = {'bottom' : {'x' : (data['left'][0], data['right'][0]),
                         'y' : (data['left'][1], data['right'][1])},
             'left' : {'x' : (data['left'][0], data['top'][0]),
                       'y' : (data['left'][1], data['top'][1])},
             'right' : {'x' : (data['top'][0], data['right'][0]),
                       'y' : (data['top'][1], data['right'][1])} }
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
    reduced_ratio = 72/(ca.amt_of_el('O'))
    Ca_ratio = ca.amt_of_el('Ca') * reduced_ratio / 9
    Na_ratio = ca.amt_of_el('Na') * reduced_ratio / 18
    pt = [Ca_ratio, Na_ratio, 1-Ca_ratio-Na_ratio]
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
    
def params():
    """
    Args:
        
    Returns:
        just colors for stable and unstable points
    """
    return {'stable' : {'c' : tableau_colors()['blue'],
                        'm' : 'o',
                        'alpha' : 1},
            'unstable' : {'c' : tableau_colors()['red'],
                          'm' : '^',
                          'alpha' : 0.2}}

def get_voltage(endpoints=[]):
    
    inside = get_triangle()
    U_test = {'Ca1O12P3V2': -7.518649259259259, 'Na1O12P3V2' : -7.291825, 'Na3O12P3V2': -6.984577666666667, 'Ca1Na6O72P18V12':-7.303014495412844,
              'Ca1Na1O12P3V2':-7.340160701754386}
    matrix = np.zeros((3,3))
    vector = np.zeros((3,1))
    for i in range(len(endpoints)):
        pt, tot = convert_comp_to_content(endpoints[i])
        matrix[i,:] = pt
        vector[i] = U_test[endpoints[i]] * tot
        # vector[i] = inside[endpoints[i]] * tot
    chempo = np.linalg.inv(matrix)@vector
    Ca_voltage = (-chempo[0][0]-2.0056)/2
    Na_voltage = (-chempo[1][0]-1.3225)/1
    Ca_chempo = chempo[0][0]
    Na_chempo = chempo[1][0]
    
    return Ca_voltage, Na_voltage, Ca_chempo, Na_chempo

def get_voltage_curve(xlim=(0, 1.0), ylim=(2.9, 3.6), 
                      xticks=(False, np.arange(0, 1.0+0.01, 0.2)), yticks=(False, np.arange(2.9, 3.61, 0.1))):
    
    set_rc_params()
    
    endpoints = ['Na1O12P3V2','Ca1Na6O72P18V12','Na3O12P3V2']
    Ca1, Na1 = get_voltage(endpoints)
    
    endpoints = ['Ca1O12P3V2', 'Na3O12P3V2', 'Na1O12P3V2']
    Ca2, Na2 = get_voltage(endpoints)
    
    endpoints = ['Ca1O12P3V2', 'Na3O12P3V2', 'Ca1Na1O12P3V2']
    Ca3, Na3 = get_voltage(endpoints)
    

    
    x = [0, 0.166, 0.666, 1.0]
    y = [Ca1, Ca2, Ca3]
    plt.ylabel(r'$Voltage(V$' + ' vs ' + r'$Ca^{2+}/Ca)$')
    plt.xlabel(r'x in $Ca_{x}NaV_{2}(PO_{4})_{3}$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.yticks(yticks[1])
    plt.xticks(xticks[1])
    for i,j in enumerate(x):
        plt.plot([x[i], x[i+1]], [y[i],y[i]], linestyle='-', color='k')
        if i < 2:
            plt.plot([x[i+1], x[i+1]], [y[i],y[i+1]], linestyle='-', color='k')
    plt.show()
        

def main():
    
    #PD_DATA_DIR = os.path.join(DATA_DIR, '0414_PD_data.json')
    #d = read_json(os.path.join(DATA_DIR, '0414_ggaU.json'))
    #d = read_json(os.path.join(DATA_DIR, '0923_10K_CE_energy.json'))
    #d = read_json(os.path.join(DATA_DIR, '1019_0K_CE_energy.json'))
    d = read_json(os.path.join(DATA_DIR, '0201_0K_DFT_energy.json'))
    converted_data = convert_data(d)
    
    total = get_total()
    inside = get_triangle()
    #hull_data, line_data = get_hullandline(converted_data, remake=True, option='inside')
    
    # For plotting total energy above hull
    hull_data, line_data = get_hullandline(total, remake=True, option='decompose')
    ax,x,y,z = trianglePD(hull_data, line_data, option='decompose')
    
    # For plotting inside only
    #hull_data, line_data = get_hullandline(converted_data, remake=True, option='inside')
    #ax,x,y,z = trianglePD(hull_data, line_data, option='decompose')
    
    return hull_data, line_data, x,y,z

if __name__ == '__main__':
    
    hull_data, line_data, x,y,z = main()
    
    
