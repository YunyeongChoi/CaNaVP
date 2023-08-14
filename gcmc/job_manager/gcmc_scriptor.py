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
from gcmc.job_manager.swift_writer import SwiftWriter
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
            self.save_path = "/home/yychoi/CaNaVP_detect/na3640ca8266"
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
            chempo_list = self.splitter(4)
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
                              'step': self.step,
                              'temp': self.temp,
                              'savepath': self.data_path,
                              'occupath': "/home/yychoi/CaNaVP/notebooks/300_Na1_occu.npy"}

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
            elif self.machine == 'swift':
                job_name = 'cn-sgmc' + str(count)
                a = SwiftWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                '/home/yychoi/CaNaVP/gcmc/launcher/basic_launcher.py',
                                python_options)
            else:
                raise ValueError('Need to specify machine correctly')

            a.write_script()
            os.chdir(path_directory)
            subprocess.call(["sbatch", "job.sh"])
            print("{} launched".format(job_name))

        return

    def temp_scan(self):
        """
        Args:
            chempo: Tuple of chemical potentials. First is Ca, Second is Na.
            temperature: List or array of temperature want to run jobs.
        Returns:
            Directories under target directory, including job scripts and launcher python file.
        Note:
        TODO: Make launcher_writer to prevent trash like codes.
        """
        assert len(self.ca_range) == 1
        assert len(self.na_range) == 1
        assert type(self.temp) == list
        assert type(self.step) == list

        chempo = (self.ca_range[0], self.na_range[0])

        if self.machine == 'eagle':
            saved_dir = '/scratch/yychoi/CaNaVP_gcMC/scan_300K/data'
        elif self.machine == 'swift':
            saved_dir = '/home/yychoi/CaNaVP_gcMC/lowU300K/data'
        else:
            raise ValueError()

        saved_key = str(chempo[0]) + '_' + str(chempo[1]) + '_occupancy.npz'
        saved_data = os.path.join(saved_dir, saved_key)
        occu_data = np.load(saved_data, 'o')
        last_occu = occu_data['o'][-1]
        occu_save_path = os.path.join(self.save_path, str(chempo[0]) + '_' + str(chempo[1]) + '.npy')
        np.save(occu_save_path, last_occu)

        for ith, t in enumerate(self.temp):

            path_directory = os.path.join(self.save_path, str(int(t)))
            if not os.path.exists(path_directory):
                os.mkdir(path_directory)

            self.data_path = os.path.join(path_directory, "data")
            if not os.path.exists(self.data_path):
                os.mkdir(self.data_path)

            python_options = {'ca_amt': 0.5,
                              'na_amt': 1.0,
                              'ca_dmu': chempo[0],
                              'na_dmu': chempo[1],
                              'step': self.step[ith],
                              'temp': t,
                              'savepath': self.data_path,
                              'occupath': occu_save_path}

            if self.machine == 'savio':
                job_name = str(chempo[0]) + '_' + str(chempo[1]) + '_' + str(t)
                a = SavioWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/global/scratch/users/yychoi94/CaNaVP/gcmc/launcher"
                                "/basic_launcher.py",
                                python_options)
            elif self.machine == 'lawrencium':
                job_name = str(chempo[0]) + '_' + str(chempo[1]) + '_' + str(t)
                a = LawrenciumWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                     "/global/scratch/users/yychoi94/CaNaVP/gcmc/launcher"
                                     "/basic_launcher_temp.py",
                                     python_options)
            elif self.machine == 'eagle':
                job_name = str(chempo[0]) + '_' + str(chempo[1]) + '_' + str(t)
                a = EagleWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/scratch/yychoi/CaNaVP/gcmc/launcher/basic_launcher.py",
                                python_options)
            elif self.machine == 'swift':
                job_name = str(chempo[0]) + '_' + str(chempo[1]) + '_' + str(t)
                a = SwiftWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                '/home/yychoi/CaNaVP/gcmc/launcher/basic_launcher.py',
                                python_options)
            else:
                raise ValueError('Need to specify machine correctly')

            a.write_script()
            os.chdir(path_directory)
            subprocess.call(["sbatch", "job.sh"])
            print("{} launched".format(job_name))

        return

    def errors(self):

        return


def main():

    """
    ca_range = np.linspace(-9.0, -5.0, 41)
    na_range = np.linspace(-4.8, -2.8, 41)
    # ca_range = np.linspace(-10.0, -9.1, 10)
    # na_range = np.linspace(-5.3, -4.85, 10)
    test = sgmcScriptor('eagle', ca_range, na_range, 10000, 300)
    test.chempo_scan(option='general')
    """
    
    """
    # For scanning chempo
    ca_range = [-7.1]
    na_range = np.linspace(-4.8, -2.81, 200)
    test = sgmcScriptor('swift', ca_range, na_range, 10000000, 300.0)
    test.chempo_scan(option='general')
    """

    # For detailed scan of phase transition
    # ca_range = np.linspace(-8.2, -5.8, 1201)
    # na_range = [-3.6]
    ca_range = np.linspace(-8.2, -6.6, 81)
    na_range = np.linspace(-4.0, -3.6, 81)
    test = sgmcScriptor('swift', ca_range, na_range, 10000000, 300.0)
    test.chempo_scan(option='general')
    
    """
    # For testing temp_scan method
    ca_range = [-5.9]
    na_range = [-3.15]
    temp_range = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0,
                  200.0, 220.0, 240.0, 260.0, 280.0]
    step_range = [10000000] * len(temp_range)
    test = sgmcScriptor('swift', ca_range, na_range, step_range, temp_range)
    test.temp_scan()
    """

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
