class NeutralChargeError(Exception):

    error_dir_list = []

    def __init__(self, msg="Total charge is nonzero"):
        self.msg = msg

    def __str__(self):
        return self.msg

    def append(self, directory):
        self.error_dir_list.append(directory)

    def get(self):
        return self.error_dir_list


class AtomMoveError(Exception):

    def __init__(self, msg="Atom moved from original position"):
        self.msg = msg

    def __str__(self):
        return self.msg
