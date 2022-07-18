"""
Collect convergence, energy, magmom, poscar, contcar.
{1 : {"concent": str, "poscar" : @pymatgen.core.Structure, "contcar" : @pymatgen.core.Structure,
"convergence": bool, "energy": float, "magmom": list}}
"""
import os
from glob import glob
from src.retriever import vasp_retriever
from src.setter import read_json, write_json
from pymatgen.core.structure import Structure


def main(calc_dir) -> None:

    # Target json file to run.
    groundjson = os.path.join(calc_dir, 'result.json')

    count = 0
    groundcount = 0
    groundlist = {}
    # Get setup files for generated folder.
    spec_list = glob(calc_dir + "/*/")
    print(spec_list)
    for i in spec_list:
        detailed_spec_list = glob(i + "*/")
        for j in detailed_spec_list:
            print(j)
            count += 1
            # Only ground state and it's HE variances.
            if str(0) in j.split("/")[-2]:
                spec = {}
                vr = vasp_retriever(os.path.join(j, 'U'))
                spec["energy"] = vr.get_energy
                spec["poscar"] = vr.get_poscar.to_json()
                spec["contcar"] = vr.get_oxidation_decorated_structure().to_json()
                spec["convergence"] = vr.is_converged
                spec["errors"] = vr.errors
                spec["directory"] = j
                groundcount += 1
                groundlist[groundcount] = spec

    write_json(groundlist, groundjson)
    print(count)

    return


if __name__ == '__main__':

    main("/global/scratch/users/yychoi94/CaNaVP/setup/calc")
