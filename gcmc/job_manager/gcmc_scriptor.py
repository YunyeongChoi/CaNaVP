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
from gcmc.job_manager.savio_writer import SavioWriter
from gcmc.job_manager.lawrencium_writer import LawrenciumWriter
from gcmc.job_manager.eagle_writer import EagleWriter
from gcmc.utils import deprecated


class sgmcScriptor:

    def __init__(self, machine, ca_range, na_range, step, temp, save_path=None):
        """
        Args:
            machine: Machine want to launch calculations.
            ca_range: np.array, array of ca chemical potential to search.
            na_range: np.array, array of na chemical potential to search.
            step: List or int
            temp: List or float.
            save_path: path that all directories will be saved.
        """
        self.machine = machine
        self.ca_range = ca_range
        self.na_range = na_range
        self.step = step
        self.temp = temp

        if save_path is None:
            self.save_path = "/scratch/yychoi/CaNaVP_gcMC/test"
        else:
            self.save_path = save_path
        if not os.path.exists(self.save_path):
            raise FileNotFoundError("Need a right path to save files")
        self.data_path = os.path.join(self.save_path, "data")
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def splitter(self, max_number):
        """
        Args:
            max_number: int, max_number of mc run to run at one node.
        Returns:
            List[List[tuple]]
        Split (ca, na) into lists that not length not exceed max_number.
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

    def match_splitter(self, max_number):
        """
        Args:
            max_number: int, max_number of mc run to run at one node.
        Returns:
            List[List[tuple]]
        match ca, na chempo list to (ca, na) tuple and split into lists that not length not
        exceed max_number.
        """
        return_list = []
        assert len(self.ca_range) == len(self.na_range)
        total = len(self.ca_range)
        count = 0
        splitted_list = []
        for i, j in enumerate(self.ca_range):
            count += 1
            if not count == total:
                if not len(splitted_list) >= max_number:
                    splitted_list.append(((np.round(j, 5)), np.round(self.na_range[i], 5)))
                else:
                    return_list.append(splitted_list)
                    splitted_list = [((np.round(j, 5)), np.round(self.na_range[i], 5))]
            else:
                splitted_list.append(((np.round(j, 5)), np.round(self.na_range[i], 5)))
                return_list.append(splitted_list)

        return return_list

    @staticmethod
    def get_jobname(options):

        line = []
        for i in options:
            line.append(str(options[i]))

        return '_'.join(line)

    def chempo_scan(self, option=''):
        """
        Write a script.
        option - general: scan all the possible combinations in the given range of Ca/Na
        option - match: scan the matched combinations in the given range of Ca/Na
        usage:
            ca_range = np.arange(-10.3, -6.3, 0.5)
            na_range = np.arange(-5.4, -3.4, 0.25)
            test = sgmcScriptor(ca_range, na_range, step, temp)
            test.main()
        """
        assert type(self.step) == int
        assert type(self.temp) == float

        if option == 'general':
            chempo_list = self.splitter(6)
        elif option == 'match':
            chempo_list = self.match_splitter(6)
        else:
            raise ValueError('not supported option')

        count = 0
        for chempo_set in chempo_list:
            count += 1
            ca_list, na_list = [], []
            for j in chempo_set:
                ca_list.append(j[0])
                na_list.append(j[1])

            if count < 10:
                path_directory = os.path.join(self.save_path, str(0) + str(count))
            else:
                path_directory = os.path.join(self.save_path, str(count))
            if not os.path.exists(path_directory):
                os.mkdir(path_directory)

            python_options = {'ca_amt': 0.5,
                              'na_amt': 1.0,
                              'ca_dmu': ca_list,
                              'na_dmu': na_list,
                              'path': self.data_path,
                              'step': self.step,
                              'temp': self.temp}
            if self.machine == 'savio':
                job_name = "cn-sgmc_" + str(count)
                a = SavioWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/global/scratch/users/yychoi94/CaNaVP/gcmc/launcher"
                                "/basic_launcher.py",
                                python_options)
            elif self.machine == 'lawrencium':
                job_name = "smol_" + str(count)
                a = LawrenciumWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                     "/global/scratch/users/yychoi94/CaNaVP/gcmc/launcher"
                                     "/basic_launcher_temp.py",
                                     python_options)
            elif self.machine == 'eagle':
                job_name = 'cn-sgmc' + str(count)
                a = EagleWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/scratch/yychoi/CaNaVP/gcmc/launcher/basic_launcher.py",
                                python_options)
            else:
                raise ValueError('Need to specify machine correctly')
            a.write_script()
            os.chdir(path_directory)
            subprocess.call(["sbatch", "job.sh"])
            print("{} launched".format(job_name))

        return

    def temp_scan(self, chempo, temperature, step=None):
        """
        Args:
            chempo: Tuple of chemical potentials. First is Ca, Second is Na.
            temperature: List or array of temperature want to run jobs.
        Returns:
            Directories under target directory, including job scripts and launcher python file.
        Note:
        TODO: Make launcher_writer to prevent trash like codes.
        """
        assert type(temperature) == List
        if step is not None:
            assert type(step) == List
        


        return

    def errors(self):

        return


def main():

    # ca_range = np.arange(-10.3, -6.3, 0.5)
    # na_range = np.arange(-5.4, -3.4, 0.25)
    # test = sgmcScriptor(ca_range, na_range)
    # test.main()

    # ca_range = np.linspace(-8.55, -6, 30)
    # na_range = [-4.525]
    # test = sgmcScriptor(ca_range, na_range)
    # test.one_cation_scan()

    # ca_range = np.arange(-8.5, -7.5, 0.1)
    # na_range = np.arange(-4.5, -3.5, 0.1)
    # test = sgmcScriptor(ca_range, na_range)
    # test.general_scan()

    # test.one_cation_scan()

    """
    ca_range = np.linspace(-9.0, -5.0, 41)
    na_range = np.linspace(-4.8, -2.8, 41)
    # ca_range = np.linspace(-10.0, -9.1, 10)
    # na_range = np.linspace(-5.3, -4.85, 10)
    test = sgmcScriptor('eagle', ca_range, na_range, 10000, 300)
    test.chempo_scan(option='general')
    """

    # For testing purpose
    ca_range = [-5.2]
    na_range = [-4.0]
    test = sgmcScriptor('eagle', ca_range, na_range, 10000, 300.0)
    test.chempo_scan(option='general')

    """
    voltage_range = np.linspace(1.5, 3.0, 100)
    ca_range, na_range = [], []
    for i in voltage_range:
        ca_voltage = -2 * i - 1.9985
        na_voltage = -i - 1.3122
        ca_range.append(np.round(ca_voltage, 3))
        na_range.append(np.round(na_voltage, 3))

    test = sgmcScriptor(ca_range, na_range)
    test.general_scan(option='match')
    """


if __name__ == '__main__':

    main()

