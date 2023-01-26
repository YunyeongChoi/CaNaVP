from gcmc.job_manager.script_writer import ScriptWriter


class LawrenciumWriter(ScriptWriter):

    def __init__(self,
                 calculation_type,
                 file_path,
                 job_name,
                 python_script_name=None,
                 python_options=None):
        """
        calculation_type: Str, python or vasp.
        filepath: Str. path to job script that will be written.
        job_name: Str. Name of job will be written in the job script.
        python_script_name: Str. python script path that will be written in te job script.
        python_options: Dict. python script options that will be written in the job script.
        TODO: Need to distinguish vasp 5.4.4 and vasp 6.3 runs.
        """

        super().__init__("lawrencium", calculation_type, file_path, job_name)
        self._account = 'lr_ceder'
        if self._account == 'lr_ceder':
            self._partition = 'lr5'
            self._qos = 'condo_ceder'
            if self._partition == 'lr5':
                self.ntasks = 28 * self.node
            self.options.pop('cpus-per-task', None)
        elif self._account == 'pc_ceder':
            self._partition = 'lr5'
            self._qos = 'lr_normal'
            self.ntasks = 28 * self.node
        self._continuous_option = True
        self.options = {"account": self._account, "partition": self._partition, "qos": self._qos}
        if self.calculation_type == 'python':
            self.python_script_name = python_script_name
            if self.python_script_name is None:
                raise ValueError("Your script path is needed as an input.")
            self.python_options = python_options
            if python_options is None:
                self.python_options = {'ca_amt': 0.5,
                                       'na_amt': 1.0,
                                       'ca_dmu': [-2, -3, -4],
                                       'na_dmu': [-3, -4, -5],
                                       'path': "/global/scratch/users/yychoi94/CaNaVP_gcMC"
                                               "/5000K_556_7584_3544"}

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
            launch_line = 'module load intel/2020.1.024.par\n'
            # Can be designate env if needed in the future.
            launch_line += 'source /global/home/users/yychoi94/miniconda3/etc/profile.d/conda.sh\n'
            launch_line += 'conda activate cn-sgmc\n'
            launch_line += 'ulimit -s unlimited\n'
            launch_line += '\n'
            launch_line += 'python {} {}> result.out\n'.format(self.python_script_name,
                                                               self.pythonoptionmaker())
        else:
            # TODO: Update for vasp 5.4.4 version.
            launch_line = 'module load intel/2020.1.024.par\n'
            launch_line += 'ulimit -s unlimited\n'
            launch_line += '\n'
            launch_line = 'mpirun -np {} //global/home/users/yychoi94/bin/lawrencium_vasp63_bin' \
                          '/vasp_std > vasp.out\n'.format(self._ntasks)

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
