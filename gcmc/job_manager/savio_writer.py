from gcmc.job_manager.script_writer import ScriptWriter


class SavioWriter(ScriptWriter):

    def __init__(self,
                 calculation_type,
                 file_path,
                 job_name,
                 python_script_name=None,
                 python_options=None):

        super().__init__("savio", calculation_type, file_path, job_name)
        self._account = 'fc_ceder'
        if self._account == 'fc_ceder':
            self._partition = 'savio3'
            self._qos = 'savio_normal'
            if self._partition == 'savio3':
                self.ntasks = 32 * self.node
            self.options.pop('cpus-per-task', None)
        elif self._account == 'co_condoceder':
            self._partition = 'savio4_htc'
            self._qos = 'condoceder_htc4_normal'
            self.ntasks = 56
            self.cpus = 1
            self.options.pop('nodes', None)
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
                                       'path': "/global/scratch/users/yychoi94/CaNaVP_gcMC/5000K_556_7584_3544"}

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
