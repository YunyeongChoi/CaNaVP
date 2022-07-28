import os
from src.setter import read_json
from plot.canvphull import set_rc_params
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.getcwd()
DATA_DIR = FIG_DIR.replace('plot', 'data')
magmom_data = read_json(os.path.join(DATA_DIR, 'magmom.json'))

if __name__ == '__main__':
    v_list = []
    set_rc_params()
    for i in magmom_data:
        ca = i.split('/')[-3].split('_')[0]
        na = i.split('/')[-3].split('_')[1]
        cations = int((float(ca) + float(na) * 6))
        v_list += magmom_data[i][cations:cations + 12]

    plt.hist(v_list, alpha=0.7, bins=200, color='k')
    plt.axvline(0.2, color='k', linestyle='--', alpha=0.2, linewidth=1)
    plt.axvline(0.8, color='k', linestyle='--', alpha=0.2, linewidth=1)
    plt.axvline(1.2, color='k', linestyle='--', alpha=0.2, linewidth=1)
    plt.axvline(1.6, color='k', linestyle='--', alpha=0.2, linewidth=1)
    plt.axvline(2.0, color='k', linestyle='--', alpha=0.2, linewidth=1)
    plt.xlabel(r"V magnetic moment ($\mu_{B}$)")
    plt.ylabel("Counts")
    plt.xlim(-0.2, 2.2)
    plt.xticks(np.arange(0, 2.2, 0.5))
    plt.title("Histogram of V moments", fontsize=20, y=1.02)
    plt.legend()
    plt.show()
