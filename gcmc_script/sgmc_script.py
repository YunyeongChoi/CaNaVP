#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2022

@author: yun
@purpose: Short script to generate job gcmc_script for submission.
@design: 1. Initialize ensemble, sampler from given ensemble.mson file.
         2. Split given range of chemical potentials.
         3. Run sampler and save sampler.sample to mson file.
"""

import os
import subprocess
import numpy as np
from abc import abstractmethod, ABCMeta


class sgmcScriptor:

    def __init__(self, ca_range, na_range):
        """
        Args:
            na_range: np.array, array of na chemical potential to search.
            ca_range: np.array, array of ca chemical potential to search.
        """
        self.ca_range = ca_range
        self.na_range = na_range
        self.base_path = "/global/scratch/users/yychoi94/CaNaVP_gcMC_300K_Naonly"

    def splitter(self, max_number):
        """
        Args:
            max_number: int, max_number of mc run to run at one node.
        Returns:
            List[List[tuple]]
        Split (na, ca) into lists that not length not exceed max_number.
        """
        return_list = []
        total = len(self.ca_range) * len(self.na_range)
        count = 0
        splitted_list = []
        for i in self.ca_range:
            for j in self.na_range:
                count += 1
                if not count == total:
                    if not len(splitted_list) >= max_number:
                        splitted_list.append((np.round(i, 5), np.round(j, 5)))
                    else:
                        return_list.append(splitted_list)
                        splitted_list = [(np.round(i, 5), np.round(j, 5))]
                else:
                    splitted_list.append((np.round(i, 5), np.round(j, 5)))
                    return_list.append(splitted_list)

        return return_list

    @staticmethod
    def get_jobname(options):

        line = []
        for i in options:
            line.append(str(options[i]))

        return '_'.join(line)

    def general_scan(self):
        """
        Write a script.
        """
        chempo_list = self.splitter(4)
        count = 0
        for chempo_set in chempo_list:
            count += 1
            ca_list, na_list = [], []
            for j in chempo_set:
                ca_list.append(j[0])
                na_list.append(j[1])

            if count < 10:
                path_directory = os.path.join(self.base_path, str(0) + str(count))
            else:
                path_directory = os.path.join(self.base_path, str(count))
            if not os.path.exists(path_directory):
                os.mkdir(path_directory)

            python_options = {'ca_amt': 0.5, 'na_amt': 1.0, 'ca_dmu': ca_list, 'na_dmu': na_list}
            # job_name = self.get_jobname(python_options)
            job_name = "cn-sgmc_" + str(count)
            a = SavioWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                            "/global/scratch/users/yychoi94/CaNaVP/gcmc_script/basic_script.py",
                            python_options)
            a.write_script()
            os.chdir(path_directory)
            subprocess.call(["sbatch", "job.sh"])
            print("{} launched".format(job_name))

        return

    def one_cation_scan(self):

        count = 0
        if len(self.ca_range) == 1:
            for chempo in self.na_range:
                count += 1
                if count < 10:
                    path_directory = os.path.join(self.base_path, str(0) + str(count))
                else:
                    path_directory = os.path.join(self.base_path, str(count))
                if not os.path.exists(path_directory):
                    os.mkdir(path_directory)
                python_options = {'ca_dmu': self.ca_range[0], 'na_dmu': chempo}
                job_name = "na_scan_" + str(count)
                a = SavioWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/global/scratch/users/yychoi94/CaNaVP/gcmc_script"
                                "/pseudo_annealing_script.py",
                                python_options)
                a.write_script()
                os.chdir(path_directory)
                subprocess.call(["sbatch", "job.sh"])
                print("{} launched".format(job_name))
        elif len(self.na_range) == 1:
            for chempo in self.ca_range:
                count += 1
                if count < 10:
                    path_directory = os.path.join(self.base_path, str(0) + str(count))
                else:
                    path_directory = os.path.join(self.base_path, str(count))
                if not os.path.exists(path_directory):
                    os.mkdir(path_directory)
                python_options = {'ca_dmu': chempo, 'na_dmu': self.na_range[0]}
                job_name = "ca_scan_" + str(count)
                a = SavioWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/global/scratch/users/yychoi94/CaNaVP/gcmc_script"
                                "/pseudo_annealing_script.py",
                                python_options)
                a.write_script()
                os.chdir(path_directory)
                subprocess.call(["sbatch", "job.sh"])
                print("{} launched".format(job_name))
        elif len(self.ca_range) > 1 and len(self.na_range) > 1:
            raise ValueError("This assume only one chemical potential change")
        else:
            raise ValueError("Set chemical potential properly")

        return

    def errors(self):

        return


class ScriptWriter(metaclass=ABCMeta):

    def __init__(self, machine, calculation_type, file_path, job_name):
        """
        Args:
            machine: Machine want to run job.
            calculation_type: Calculation want to run.
        """
        self._machine = machine
        self._calculation_type = calculation_type
        self._file_path = file_path
        self._job_name = job_name
        self._node = 1
        self._ntasks = 20
        self._walltime = "24:00:00"
        self._err_file = "log.e"
        self._out_file = "log.o"
        self._options = {'nodes': self._node, 'ntasks': self._ntasks, 'output': self._out_file,
                         'error': self._err_file, 'time': self._walltime,
                         'job-name': self._job_name}

        # if calculation_type = python, need basic python file. - Will be done in basic_script.py
        # need to address script name want to run in python case.
        # change src.setter to this one.
        # need to address path contradiction.

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, name):
        if name not in ["savio", "cori", "stampede", "bridges"]:
            raise MachineNameError("Not in available machine list. choose one of savio, cori, "
                                   "stampede, bridges.")
        self._machine = name

    @property
    def calculation_type(self):
        return self._calculation_type

    @calculation_type.setter
    def calculation_type(self, cal_type):
        if cal_type not in ["DFT_basic", "NEB", "AIMD", "python"]:
            raise CalculationTypeError("Not in availabe calculation types. choose one of DFT_basic,"
                                       "NEB, AIMD, python.")
        self._calculation_type = cal_type

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, option):
        """
        This is updating new option to self.options. Not a reset.
        """
        for key in option:
            self.options[key] = option[key]

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, path):
        self._file_path = path

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node_number):
        self._node = node_number
        self.options = {"nodes": node_number}

    @property
    def ntasks(self):
        return self._ntasks

    @ntasks.setter
    def ntasks(self, ntasks_number):
        self._ntasks = ntasks_number
        self.options = {"ntasks": ntasks_number}

    @property
    def walltime(self):
        return self._walltime

    @walltime.setter
    def walltime(self, caltime):
        self._walltime = caltime
        self.options = {"walltime": caltime}

    @property
    def err_file(self):
        return self._err_file

    @err_file.setter
    def err_file(self, err_file_name):
        self._err_file = err_file_name
        self.options = {"error": err_file_name}

    @property
    def out_file(self):
        return self._out_file

    @out_file.setter
    def out_file(self, out_file_name):
        self._out_file = out_file_name
        self.options = {"out": out_file_name}

    @abstractmethod
    def pythonoptionmaker(self):
        """
        Append options to end of the python executing line.
        """

    @abstractmethod
    def punchline(self):
        """
        Line for execute script.
        """
        return

    @abstractmethod
    def write_script(self):
        """
        Write a script in a target directory.
        """
        return


class SavioWriter(ScriptWriter):

    def __init__(self, calculation_type, file_path, job_name, python_script_name=None,
                 python_options=None):

        super().__init__("savio", calculation_type, file_path, job_name)
        self._account = 'fc_ceder'
        self._partition = 'savio3'
        self._qos = 'savio_normal'
        if self._partition == 'savio3':
            self.ntasks = 32 * self.node
        self._continuous_option = True
        self.options = {"account": self._account, "partition": self._partition, "qos": self._qos}
        if self.calculation_type == 'python':
            self.python_script_name = python_script_name
            if self.python_script_name is None:
                raise ValueError("Your script path is needed as an input.")
            self.python_options = python_options
            if python_options is None:
                self.python_options = {'ca_amt': 0.5, 'na_amt': 1.0, 'ca_dmu': [-2, -3, -4],
                                       'na_dmu': [-3, -4, -5]}

    @property
    def account(self):
        return self._account

    @account.setter
    def account(self, account_name):
        self._account = account_name

    @property
    def partition(self):
        return self._partition

    @partition.setter
    def partition(self, partition_name):
        self._partition = partition_name

    @property
    def qos(self):
        return self._qos

    @qos.setter
    def qos(self, qos_name):
        self._qos = qos_name

    def pythonoptionmaker(self):

        line = ''
        for i in self.python_options:
            if type(self.python_options[i]) is not list:
                line += '--' + i + ' ' + str(self.python_options[i]) + ' '
            else:
                line += '--' + i + ' '
                for j in self.python_options[i]:
                    line += str(j) + ' '

        return line

    def punchline(self):

        if self.calculation_type == "python":
            launch_line = 'module load python/3.9.12\n'
            # Can be designate env if needed in the future.
            launch_line += 'source activate cn-sgmc\n'
            launch_line += 'python {} {}> result.out\n'.format(self.python_script_name,
                                                               self.pythonoptionmaker())
        else:
            launch_line = 'mpirun -n {} /global/home/users/yychoi94/bin/vasp.5.4.4_vtst178_' \
                          'with_DnoAugXCMeta/vasp_std > vasp.out\n'.format(self._ntasks)

        return launch_line

    def dftline(self):

        if self._continuous_option:
            line = 'mkdir U; cd U;\n'
            line += "IsConv=`grep 'required accuracy' OUTCAR`;\n"
            line += 'if [ -z "${IsConv}" ]; then\n'
            line += '    if [ -s "CONTCAR" ]; then cp CONTCAR POSCAR; fi;\n'
            line += '    if [ ! -s "POSCAR" ]; then\n'
            line += '        cp ../{KPOINTS,POTCAR,POSCAR} .;\n'
            line += '    fi\n'
            line += '    cp ../INCAR .;\n'
            line += '    ' + self.punchline() + '\n'
            line += 'fi'
        else:
            line = 'mkdir U; cd U;\n'
            line += 'cp ../{KPOINTS,POTCAR,POSCAR,INCAR} .;\n'

        return line

    def write_script(self):

        line1 = '#!/bin/bash\n'
        with open(self.file_path, 'w') as f:
            f.write(line1)
            for tag in self.options:
                option = self.options[tag]
                if option:
                    option = str(option)
                    f.write('%s --%s=%s\n' % ('#SBATCH', tag, option))
            f.write('\n')

            if self.calculation_type in ["DFT", "NEB", "AIMD"]:
                line = self.dftline()
                f.write(line)
            elif self.calculation_type in ["python"]:
                line = self.punchline()
                f.write(line)
            else:
                pass

            f.close()

        return


"""
Errors will be separated.
"""


class MachineNameError(Exception):

    def __init__(self, msg="Do not support that machine"):
        self.msg = msg

    def __str__(self):
        return self.msg


class CalculationTypeError(Exception):

    error_dir_list = []

    def __init__(self, msg="Do not support that calculation"):
        self.msg = msg

    def __str__(self):
        return self.msg


def main():

    # ca_range = np.arange(-10.3, -6.3, 0.5)
    # na_range = np.arange(-5.4, -3.4, 0.25)
    # test = sgmcScriptor(ca_range, na_range)
    # test.main()

    # ca_range = np.linspace(-8.55, -6, 30)
    # na_range = [-4.525]
    # test = sgmcScriptor(ca_range, na_range)
    # test.one_cation_scan()

    ca_range = [-8.55]
    na_range = np.linspace(-4.525, -3.3, 30)
    test = sgmcScriptor(ca_range, na_range)
    test.one_cation_scan()

if __name__ == '__main__':

    main()
