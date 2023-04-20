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
from glob import glob
from gcmc.job_manager.savio_writer import SavioWriter
from gcmc.job_manager.lawrencium_writer import LawrenciumWriter
from gcmc.job_manager.eagle_writer import EagleWriter
from gcmc.job_manager.swift_writer import SwiftWriter
from gcmc.utils import deprecated


class kmcScriptor:

    def __init__(self, machine, step, temp, occu_dir=None, save_path=None):
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
        self.step = step
        self.temp = temp

        if occu_dir is None:
            self.occu_dir = "/scratch/yychoi/CaNaVP_gcMC/rough_scan_300K"
        else:
            self.occu_dir = occu_dir
        if not os.path.exists(self.occu_dir):
            raise FileNotFoundError("Occupancy directory is not exists.")

        if save_path is None:
            self.save_path = "/scratch/yychoi/CaNaVP_gcMC/kmc_test"
        else:
            self.save_path = save_path
        if not os.path.exists(self.save_path):
            raise FileNotFoundError("Need a right path to save files")
        self.data_path = os.path.join(self.save_path, "data")
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    @staticmethod
    def get_jobname(options):

        line = []
        for i in options:
            line.append(str(options[i]))

        return '_'.join(line)

    def basic_run(self):

        occu_data_dir = os.path.join(self.occu_dir, "data")
        saved_list = glob(occu_data_dir + "/*")
        occu_path_list = []
        for i in saved_list:
            if 'occupancy.npz' in i:
                occu_path_list.append(i)
        
        count = 0
        occu_path_list.sort()
        splitted_list = [occu_path_list[i:i + 6] for i in range(0, len(occu_path_list), 6)]

        for j in splitted_list:

            count += 1
            savepath_list = []

            for k in j:
                name = k.split('/')[-1]
                key = (name.split("_")[0] + '_' + name.split("_")[1])
                savename = os.path.join(self.data_path, key + '.json')
                savepath_list.append(savename)

            python_options = {'occu': j,
                              'savepath': savepath_list,
                              'nsteps': self.step,
                              't': self.temp}

            if count < 10:
                path_directory = os.path.join(self.save_path, str(0) + str(count))
            else:
                 path_directory = os.path.join(self.save_path, str(count))
            if not os.path.exists(path_directory):
                os.mkdir(path_directory)

            if self.machine == 'savio':
                job_name = "kmc_" + str(count)
                a = SavioWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/global/scratch/users/yychoi94/CaNaVP/gcmc/launcher"
                                "/basic_launcher.py",
                                python_options)
            elif self.machine == 'lawrencium':
                job_name = "kmc_" + str(count)
                a = LawrenciumWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                     "/global/scratch/users/yychoi94/CaNaVP/gcmc/launcher"
                                     "/basic_launcher_temp.py",
                                     python_options)
            elif self.machine == 'eagle':
                job_name = 'kmc' + str(count)
                a = EagleWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                "/scratch/yychoi/CaNaVP/kmc/kmc.py",
                                python_options)
            elif self.machine == 'swift':
                job_name = 'kmc' + str(count)
                a = SwiftWriter("python", os.path.join(path_directory, 'job.sh'), job_name,
                                '/home/yychoi/CaNaVP/gcmc/launcher/basic_launcher.py',
                                python_options)
            else:
                raise ValueError('Need to specify machine correctly')

            a.write_script()
            os.chdir(path_directory)
            # subprocess.call(["sbatch", "job.sh"])
            # print("{} launched".format(job_name))

        return


def main():

    a = kmcScriptor("eagle", 10000, 300)
    a.basic_run()


if __name__ == '__main__':

    main()

