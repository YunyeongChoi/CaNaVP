import os
import warnings
import argparse
from glob import glob
from subprocess import call
from dft.setter import PoscarGen, InputGen, read_json


class launcher(object):

    def __init__(self, machine, hpc, option, input, continuous, calc_dir=None):

        """
        Args:
            machine: Machine to make input files.
            hpc: Machine to run jobs.
            option: Convergence option. fast or full.
            input: Input option. If true make input files again.
            calc_dir: Calculation directory.

        # Need to be rewritten. Very dirty structure.
        """

        self.machine = machine
        if calc_dir is None:
            if self.machine in ['savio', 'lawrencium']:
                self.calc_dir = '/global/scratch/users/yychoi94/CaNaVP_DFT_rigorous'
            elif self.machine == 'cori':
                self.calc_dir = '/global/cscratch1/sd/yychoi/JCESR/CaNaVP_DFT'
            elif self.machine == 'stampede2':
                self.calc_dir = '/scratch/06991/tg862905/JCESR/CaNaVP_DFT'
            elif self.machine == 'YUN':
                self.calc_dir = '/Users/yun/Desktop/github_codes/CaNaVP_DFT'
            elif self.machine == 'bridges2':
                self.calc_dir = '/ocean/projects/dmr060032p/yychoi/CaNaVP_DFT'
            elif self.machine == 'eagle':
                self.calc_dir = '/scratch/yychoi/CaNaVP_DFT_02'
            else:
                warnings.warn("Check machine option", DeprecationWarning)
            if not os.path.exists(self.calc_dir):
                warnings.warn("Running in the wrong machine", DeprecationWarning)
        else:
            self.calc_dir = calc_dir

        if hpc not in ['savio', 'cori', 'stampede2', 'bridges2', 'lawrencium']:
            warnings.warn("Check hpc option", DeprecationWarning)
        self.hpc = hpc

        if option not in ['fast', 'full']:
            warnings.warn("Check calculation option", DeprecationWarning)
        self.option = option

        if input is not bool:
            warnings.warn("Check input option", DeprecationWarning)
        self.input = input

        if continuous is not bool:
            warnings.warn("Check input option", DeprecationWarning)
        self.continuous = continuous

        # Target json file to run.
        # self.resultjson = read_json(os.path.join(self.calc_dir, 'result.json'))

    def poscar_setter(self) -> None:

        if not self.input:
            print("Setting POSCARs...")
            poscarrun = PoscarGen(self.input)
            # Get ground state POSCAR.
            poscarrun.run()
            # Get HE state POSCAR.
            poscarrun.HEstaterun()

        return

    def launch_jobs(self) -> None:

        count = 0
        spec_list = glob(self.calc_dir + "/*/")
        for i in spec_list:
            detailed_spec_list = glob(i + "*/")
            for j in detailed_spec_list:
                # Only ground state and it's HE variances.
                if str(2) in j.split("/")[-2]:
                    count += 1
                    os.chdir(j)
                    inputgenerator = InputGen(self.machine,
                                              self.hpc,
                                              j,
                                              self.option,
                                              True)
                    inputgenerator.at_once()
                    call(['sbatch', 'job.sh'])
                    print("{} launched".format(j))

        print("total {} launched.".format(count))

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=False, default='YUN',
                        help="Machine that want to run this python file. Yun, cori, stampede2, "
                             "bridges2, savio, lawrencium are available.")
    parser.add_argument('-p', type=str, required=False, default='savio',
                        help="HPC that want to run DFT calculation. cori, stampede2, "
                             "bridges2, savio, lawrencium are available.")
    parser.add_argument('-o', type=str, required=False, default='fast',
                        help="Option for DFT calculation. fast or full.")
    parser.add_argument('-i', type=bool, required=False, default=True,
                        help="Option for input generation. If true, only input generator runs."
                             "If false, only poscar_setter runs.")
    parser.add_argument('-l', type=bool, required=False, default=True,
                        help="Option for run jobs. If true, runs.")
    parser.add_argument('-c', type=bool, required=False, default=True,
                        help="Option for continuous job or relaunch. If true run continuous job,"
                             "else, relaunch from start.")
    args = parser.parse_args()

    # Need to test argument parser type checker.
    lj = launcher(args.m, args.p, args.o, args.i, args.c)
    # lj.poscar_setter()
    if args.l:
        lj.launch_jobs()
