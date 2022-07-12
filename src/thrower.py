import os
import warnings
import argparse
from glob import glob
from subprocess import call
from pymatgen.core.structure import Structure
from src.setter import PoscarGen, InputGen, write_json, read_json


def main(machine, hpc, option, inputoption) -> None:
    calc_dir = ''
    if machine == 'savio':
        calc_dir = '/global/scratch/users/yychoi94/CaNaVP/setup'
    elif machine == 'cori':
        calc_dir = '/global/cscratch1/sd/yychoi/JCESR/CaNaVP/setup'
    elif machine == 'stampede2':
        calc_dir = '/scratch/06991/tg862905/JCESR/CaNaVP/setup'
    elif machine == 'YUN':
        calc_dir = '/Users/yun/Desktop/github_codes/CaNaVP/setup'
    elif machine == 'bridges2':
        calc_dir = '/ocean/projects/dmr060032p/yychoi/CaNaVP/setup'
    else:
        warnings.warn("Check machine option", DeprecationWarning)

    if not os.path.exists(calc_dir):
        warnings.warn("Running in the wrong machine", DeprecationWarning)

    if hpc not in ['savio', 'cori', 'stampede2', 'bridges2']:
        warnings.warn("Check hpc option", DeprecationWarning)

    calc_dir = os.path.join(calc_dir, 'calc')
    fjson = os.path.join(calc_dir, 'calc_list.json')
    groundjson = os.path.join(calc_dir, 'ground_list.json')

    if not inputoption:
        poscarrun = PoscarGen(calc_dir)
        # Get ground state POSCAR.
        poscarrun.run()

    if not inputoption:
        # Get HE state POSCAR.
        poscarrun.HEstaterun()

    count = 0
    groundcount = 0
    calclist = {}
    groundlist = {}
    # Get setup files for generated folder.
    spec_list = glob(calc_dir + "/*/")
    for i in spec_list:
        detailed_spec_list = glob(i + "*/")
        for j in detailed_spec_list:
            count += 1
            calclist[count] = j
            if str(0) in j.split("/")[-2]:
                groundcount += 1
                groundlist[groundcount] = j
            inputgenerator = InputGen(machine, hpc, j, option)
            inputgenerator.at_once()

    write_json(calclist, fjson)
    write_json(groundlist, groundjson)
    print(count)

    return


def launchjobs(machine) -> None:

    calc_dir = ''
    if machine == 'savio':
        calc_dir = '/global/scratch/users/yychoi94/CaNaVP/setup'
    elif machine == 'cori':
        calc_dir = '/global/cscratch1/sd/yychoi/JCESR/CaNaVP/setup'
    elif machine == 'stampede2':
        calc_dir = '/scratch/06991/tg862905/JCESR/CaNaVP/setup'
    elif machine == 'YUN':
        calc_dir = '/Users/yun/Desktop/github_codes/CaNaVP/setup'
    elif machine == 'bridges2':
        calc_dir = '/ocean/projects/dmr060032p/yychoi/CaNaVP/setup'
    else:
        warnings.warn("Check machine option", DeprecationWarning)

    calc_dir = os.path.join(calc_dir, 'calc')
    groundjson = read_json(os.path.join(calc_dir, 'ground_list.json'))

    for i in groundjson:
        if int(i) < 120:
            os.chdir(groundjson[i])
            call(['sbatch', 'job.sh'])
            print("{} launched".format(groundjson[i]))

    return


def changelatticevector():
    # Will be deleted.
    from pymatgen.core.lattice import Lattice

    test_poscar = "/Users/yun/Desktop/github_codes/CaNaVP/setup/calc/0.167_2.0/0/POSCAR"
    test_structure = Structure.from_file(test_poscar)
    test_lattice = Lattice([[16.7288, 0., 0.],
                            [-4.3644, 7.559363, 0.],
                            [0., 0., 21.8042]])

    new_structure = Structure(test_lattice, test_structure.species,
                              test_structure.frac_coords,
                              test_structure.charge, False, False, False, None)

    return new_structure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=False, default='YUN',
                        help="Machine that want to run this python file. Yun, cori, stampede2, "
                             "bridges2, savio are available.")
    parser.add_argument('-p', type=str, required=False, default='savio',
                        help="HPC that want to run DFT calculation. cori, stampede2, "
                             "bridges2, savio are available.")
    parser.add_argument('-o', type=str, required=False, default='fast',
                        help="Option for DFT calculation. fast or full.")
    parser.add_argument('-i', type=bool, required=False, default=True,
                        help="Option for input generation. If true, only input generator runs.")
    parser.add_argument('-l', type=bool, required=False, default=True,
                        help="Option for run jobs. If true, runs.")
    args = parser.parse_args()

    main(args.m, args.p, args.o, args.i)

    if args.l:
        launchjobs(args.m)