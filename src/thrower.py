import os
import warnings
import argparse
from glob import glob
from subprocess import call
from pymatgen.core.structure import Structure
from src.setter import PoscarGen, InputGen, write_json, read_json


class launcher(object):

    def __init__(self, machine, hpc, option, input, calc_dir=None):

        """
        :param machine: Machine to make input files.
        :param hpc: Machine to run jobs.
        :param option: Convergence option. fast or full.
        :param input: Input option. If true make input files again.
        :param calc_dir: Calculation directory.
        """

        self.machine = machine
        if calc_dir is None:
            if self.machine == 'savio':
                self.calc_dir = '/global/scratch/users/yychoi94/CaNaVP/setup'
            elif self.machine == 'cori':
                self.calc_dir = '/global/cscratch1/sd/yychoi/JCESR/CaNaVP/setup'
            elif self.machine == 'stampede2':
                self.calc_dir = '/scratch/06991/tg862905/JCESR/CaNaVP/setup'
            elif self.machine == 'YUN':
                self.calc_dir = '/Users/yun/Desktop/github_codes/CaNaVP/setup'
            elif self.machine == 'bridges2':
                self.calc_dir = '/ocean/projects/dmr060032p/yychoi/CaNaVP/setup'
            else:
                warnings.warn("Check machine option", DeprecationWarning)
            if not os.path.exists(self.calc_dir):
                warnings.warn("Running in the wrong machine", DeprecationWarning)
        else:
            self.calc_dir = calc_dir

        if hpc not in ['savio', 'cori', 'stampede2', 'bridges2']:
            warnings.warn("Check hpc option", DeprecationWarning)
        self.hpc = hpc

        if option not in ['fast', 'full']:
            warnings.warn("Check calculation option", DeprecationWarning)
        self.option = option

        if input is not bool:
            warnings.warn("Check input option", DeprecationWarning)
        self.input = input

        self.calc_dir = os.path.join(self.calc_dir, 'calc')
        self.fjson = read_json(os.path.join(self.calc_dir, 'calc_list.json'))
        # Target json file to run.
        self.groundjson = read_json(os.path.join(calc_dir, 'ground_list.json'))

    def main(self) -> None:

        if not self.input:
            poscarrun = PoscarGen(self.input)
            # Get ground state POSCAR.
            poscarrun.run()
            # Get HE state POSCAR.
            poscarrun.HEstaterun()

        count = 0
        groundcount = 0
        calclist = {}
        groundlist = {}
        # Get setup files for generated folder.
        spec_list = glob(self.calc_dir + "/*/")
        for i in spec_list:
            detailed_spec_list = glob(i + "*/")
            for j in detailed_spec_list:
                count += 1
                calclist[count] = j
                if str(0) in j.split("/")[-2]:
                    groundcount += 1
                    groundlist[groundcount] = j
                inputgenerator = InputGen(self.machine, self.hpc, j, self.option)
                inputgenerator.at_once()

        write_json(calclist, self.fjson)
        write_json(groundlist, self.groundjson)
        print(count)

        return

    def launchjobs(self) -> None:

        for i in self.groundjson:
            if int(i) < 120:
                os.chdir(self.groundjson[i])
                call(['sbatch', 'job.sh'])
                print("{} launched".format(self.groundjson[i]))

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

    lj = launcher(args.m, args.p, args.o, args.i)
    lj.launchjobs()

    if args.l:
        launchjobs(args.m)
