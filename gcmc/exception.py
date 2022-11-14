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