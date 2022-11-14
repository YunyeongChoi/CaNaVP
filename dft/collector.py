"""
Collect convergence, energy, magmom, poscar, contcar.
{1 : {"concent": str, "poscar" : @pymatgen.core.Structure, "contcar" : @pymatgen.core.Structure,
"convergence": bool, "energy": float, "magmom": list}}
"""
import os
from glob import glob
from dft.retriever import vasp_retriever
from dft.setter import write_json


def main(calc_dir) -> None:

    # Target json file to run.
    groundjson = os.path.join(calc_dir, 'result_1s.json')

    count = 0
    groundcount = 0
    groundlist = {}
    # Get setup files for generated folder.
    spec_list = glob(calc_dir + "/*/")
    for i in spec_list:
        detailed_spec_list = glob(i + "*/")
        for j in detailed_spec_list:
            # Only ground state and it's HE variances.
            if str(1) in j.split("/")[-2]:
                count += 1
                print(j)
                spec = {}
                vr = vasp_retriever(os.path.join(j, 'U'))
                spec["convergence"] = vr.is_converged
                spec["energy"] = vr.get_energy
                spec["poscar"] = vr.get_poscar.to_json()
                try:
                    spec["contcar"] = vr.get_oxidation_decorated_structure().to_json()
                except ValueError:
                    spec["contcar"] = "Empry contcar"
                spec["setting"] = vr.get_setting()
                spec["errors"] = vr.errors
                spec["directory"] = j
                groundcount += 1
                groundlist[groundcount] = spec

    write_json(groundlist, groundjson)
    print(count)

    return


def get_all_magmoms(calc_dir) -> None:
    """
    Just collecting all the magnetic moments of V.
    """
    magmomjson = os.path.join(calc_dir, 'magmom.json')
    magmomdict = {}

    # Get setup files for generated folder.
    spec_list = glob(calc_dir + "/*/")
    for i in spec_list:
        detailed_spec_list = glob(i + "*/")
        for j in detailed_spec_list:
            # Only ground state and it's HE variances.
            if str(0) in j.split("/")[-2]:
                print(j)
                vr = vasp_retriever(os.path.join(j, 'U'))
                magmomlist = vr.get_magnetic_moment()
                magmomdict[j] = magmomlist

    write_json(magmomdict, magmomjson)


if __name__ == '__main__':

    main("/global/scratch/users/yychoi94/CaNaVP/setup/calc")
    # get_all_magmoms("/global/scratch/users/yychoi94/CaNaVP/setup/calc")
