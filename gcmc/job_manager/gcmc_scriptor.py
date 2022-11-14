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
