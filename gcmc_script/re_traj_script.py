import os
import json
import numpy as np
from glob import glob
from smol.cofe.space import Vacancy
from pymatgen.core.sites import Species
from smol.moca.sampler.container import SampleContainer


def read_json(fjson):

    with open(fjson) as f:
        return json.load(f)


def write_json(d, fjson):

    with open(fjson, 'w') as f:
        json.dump(d, f)
    # return d


def species():

    na = Species('Na', 1)
    ca = Species('Ca', 2)
    v3 = Species('V', 3)
    v4 = Species('V', 4)
    v5 = Species('V', 5)
    vac = Vacancy()

    return ca, na, v3, v4, v5, vac


def main():

    ca, na, v3, v4, v5, vac = species()
    json_save_dir = "/global/scratch/users/yychoi94/CaNaVP/gcmc_script/traj_data.json"
    save_dir = "/global/scratch/users/yychoi94/CaNaVP_gcMC/data"
    saved_list = glob(save_dir + "/*")
    # Currently have a problem on saving.
    # saved_list.remove("/global/scratch/users/yychoi94/CaNaVP_gcMC/data/-11.0_-2.0_cn_sgmc.mson")

    if os.path.exists(json_save_dir):
        result_dict = read_json(json_save_dir)
    else:
        result_dict = {}

    for end_list in saved_list:
        print(end_list)
        name = '_'.join(end_list.split('/')[-1].split('_')[:2])
        if not name in result_dict.keys():
            temp = SampleContainer.from_hdf5(end_list)
            energy_set = temp.get_energies()
            species_count = temp.get_species_counts()
            stop_i = len(energy_set)
            for i, j in enumerate(energy_set):
                if (np.var(energy_set[i:i+1000])) < 0.1:
                    stop_i = i
                    break
            result_dict[name] = {}
            result_dict[name]['Na'] = list(species_count[na][:stop_i])
            result_dict[name]['Ca'] = list(species_count[ca][:stop_i])
            result_dict[name]['Na_last'] = species_count[na][-1]
            result_dict[name]['Ca_last'] = species_count[ca][-1]
            result_dict[name]['energy'] = list(energy_set[:stop_i])
            result_dict[name]['energy_last'] = energy_set[-1]
            result_dict[name]['metadata'] = temp.metadata

        write_json(result_dict, json_save_dir)

    return


if __name__ == '__main__':

    main()
