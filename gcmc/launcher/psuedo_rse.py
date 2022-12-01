"""
Things working Like Replica Set Exchange Monte Carlo.
300K, 5000K will be repeated to get the minimum energy V ordering.
Args: Temperature (two currently)
      Steps for each temperature (Heuristic currently)
"""

from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from copy import deepcopy
import numpy as np
import warnings
import os

__setupdir__ = '/Users/yun/Desktop/github_codes/CaNaVP/setup'
__basestructure__ = os.path.join(__setupdir__, "primPOSCAR")

# Example code for RSE

class NasiconSite(PeriodicSite):
    """
    Inherit Pymatgen.core.sites.PeriodicSite class.
    Adding neighbor, site information in base NASICON lattice.
    Args:
        site: Pymatgen.core.sites.PeriodicSite Object want to change as NascionSite Object.
        basestructure: Base NASICON structure want to use. If None get the basic NASICON structure
        from setup directory. basestructure need to have Na in all available cation sites, and V in
        all available TM sites. No oxidation decoration on Na and V.
    """

    def __init__(self, site, basestructure=None):

        super().__init__(site.species, site.frac_coords, site.lattice)
        if basestructure is None:
            self.basestructure = Structure.from_file(__basestructure__)
        else:
            self.basestructure = basestructure
        self.neighbor = self.find_neigbors()
        self.isbsite = self.classify_site()

    def find_neigbors(self) -> list:

        neighborlist = []
        if self.specie.name == "Na":
            for iteration in self.basestructure:
                if 3 < iteration.distance(self) < 3.4 and iteration.specie.name == "Na":
                    neighborlist.append(iteration)
        else:
            neighborlist = None

        return neighborlist

    def get_neighbors(self) -> list:

        return self.neighbor

    def classify_site(self):

        if self.specie.name != "Na":
            return None
        elif len(self.neighbor) == 2 and self.specie.name == "Na":
            return False
        elif len(self.neighbor) == 6 and self.specie.name == "Na":
            return True
        else:
            warnings.warn("Site is not classifies well - check data", DeprecationWarning)


class NasiconStructure(Structure):
    """
    Inherite pymatgen.core.structure.Structure class.
    classfy cations in structures to 6b, 18e sites.
    Each site is NasiconSite object.
    TODO test classification on more structure.
    TODO Need to clean siteinfo. There'll be cleaner way.
    Args:
        structure: pymatgen.core.structure.Structure Object want to change as NasiconStructure
        Object.
        basestructure: Base NASICON structure want to use. If None get the basic NASICON structure
        from setup directory.
    """

    def __init__(self, structure, basestructure=None):

        super().__init__(structure.lattice, structure.species, structure.frac_coords,
                         structure.charge, False, False, False, None)

        self.siteinfo = {'b': {'Ca': {}, 'Na': {}},
                         'e': {'Ca': {}, 'Na': {}},
                         'v': {}}
        if basestructure is None:
            self.basestructure = Structure.from_file(__basestructure__)
        else:
            self.basestructure = basestructure
        self.update_sites(structure)
        # self.classify_sites()

    def update_sites(self, structure):

        a_ratio = int(structure.lattice.a / self.basestructure.lattice.a)
        b_ratio = int(structure.lattice.b / self.basestructure.lattice.b)
        c_ratio = int(structure.lattice.c / self.basestructure.lattice.c)

        self.basestructure.make_supercell([a_ratio, b_ratio, c_ratio])
        # This is same as self.basestructure, but sites are not NasiconSites object.
        # Very lazy way.
        retained_structure = deepcopy(self.basestructure)

        # Need to update supercell case for NasiconSite class
        for i, j in enumerate(self.basestructure):
            nasiconinfo = NasiconSite(j, retained_structure)
            self.basestructure[i] = nasiconinfo

        # Updating PerodicSite objects to NasiconSite objects.
        for i, site in enumerate(structure):
            for j, base_site in enumerate(self.basestructure):
                if np.allclose(site.frac_coords, base_site.frac_coords):
                    self[i] = NasiconSite(self[i], retained_structure)
                    if base_site.isbsite:
                        if site.specie.name == 'Ca':
                            self.siteinfo['b']['Ca'][i] = (j, base_site.frac_coords)
                        elif site.specie.name == 'Na':
                            self.siteinfo['b']['Na'][i] = (j, base_site.frac_coords)
                        else:
                            warnings.warn("None-cation element placed in 6b site",
                                          DeprecationWarning)
                    elif base_site.isbsite is None and base_site.specie.name == 'V':
                        self.siteinfo['v'][i] = (j, base_site.frac_coords)
                    elif base_site.isbsite is None:
                        pass
                    elif not base_site.isbsite:
                        if site.specie.name == 'Ca':
                            self.siteinfo['e']['Ca'][i] = (j, base_site.frac_coords)
                        elif site.specie.name == 'Na':
                            self.siteinfo['e']['Na'][i] = (j, base_site.frac_coords)
                        else:
                            warnings.warn("None-cation element placed in 18e site",
                                          DeprecationWarning)
                    else:
                        warnings.warn("Not well classified sites", DeprecationWarning)


def test():

    test_structure = Structure.from_file("/Users/yun/Desktop/github_codes/CaNaVP/setup/testPOSCAR")
    test_case = NasiconStructure(test_structure)
    print(test_case.siteinfo)

    return

if __name__ == "__main__":

    test()