from abc import abstractmethod, ABCMeta
from gcmc.exception import MachineNameError, CalculationTypeError


class ScriptWriter(metaclass=ABCMeta):

    def __init__(self,
                 machine,
                 calculation_type,
                 file_path,
                 job_name):
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
        self._cpus = 1
        self._walltime = "24:00:00"
        self._err_file = "log.e"
        self._out_file = "log.o"
        self._options = {'nodes': self._node,
                         'ntasks-per-node': self._ntasks,
                         'cpus-per-task': self._cpus,
                         'output': self._out_file,
                         'error': self._err_file,
                         'time': self._walltime,
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
        self.options = {"ntasks-per-node": ntasks_number}

    @property
    def cpus(self):
        return self._cpus

    @cpus.setter
    def cpus(self, cpu_number):
        self._cpus = cpu_number
        self.options = {"cpus-per-task": cpu_number}

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
