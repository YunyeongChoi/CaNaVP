import json
import functools
import warnings
import numpy as np
from copy import deepcopy


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return wrapper


def get_dim_ids_by_sublattice(bits):
    """
    Get the component index of each species in vector n.

    Args:
        bits(List[List[Specie|Vacancy|Element]]):
           Species on each sub-lattice.
    Returns:
        Component index of each species on each sublattice in vector n:
           List[List[int]]
    """
    dim_ids = []
    dim_id = 0
    for species in bits:
        dim_ids.append(list(range(dim_id, dim_id + len(species))))
        dim_id += len(species)
    return dim_ids


def flip_vec_to_reaction(u, bits):
    """
    Convert flip direction into a reaction formula in string.

    This function is for easy interpretation of flip directions.
    Args:
        u(1D ArrayLike[int]):
            The flip vector in number change of species.
        bits(List[List[Specie|DummySpecie|Element|Vacancy]]):
            Species on all sub-lattices.
    Return:
        Reaction formulas: str.
    """
    u = np.array(u, dtype=int)
    dim_ids = get_dim_ids_by_sublattice(bits)

    from_strs = []
    to_strs = []

    for sl_id, (sl_species, sl_dims) in enumerate(zip(bits, dim_ids)):
        for specie, dim in zip(sl_species, sl_dims):
            if u[dim] < 0:
                from_strs.append('{} {}({})'
                                 .format(-u[dim], str(specie), sl_id))
            elif u[dim] > 0:
                to_strs.append('{} {}({})'
                               .format(u[dim], str(specie), sl_id))

    from_str = ' + '.join(from_strs)
    to_str = ' + '.join(to_strs)
    return from_str + ' -> ' + to_str


def divide_matrix(matrix):
    """
    To divide supercell matrix into two.

    Args:
        matrix:
            np.ndarray of 3x3 matrix
    Returns:
        shortest elongated matrix and the residual parts
    """
    diag = matrix.diagonal()
    max_position = np.argmax(matrix.diagonal())
    in_diag = deepcopy(diag)
    out_diag = deepcopy(diag)
    for i, j in enumerate(in_diag):
        if i == max_position:
            in_diag[i] = j / 2
        else:
            in_diag[i] = 1
    for i, j in enumerate(out_diag):
        if i == max_position:
            out_diag[i] = 2
        else:
            pass

    return np.diag(in_diag), np.diag(out_diag)
