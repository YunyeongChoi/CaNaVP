# General import
import time
import argparse
import numpy as np
from copy import deepcopy

# smol import
from smol.moca import Sampler

# from in house module
from dft.setter import PoscarGen
from gcmc.utils import flip_vec_to_reaction
from gcmc.launcher.base_launcher import gcmcabc
from gcmc.utils import divide_matrix


class gcmc_basic(gcmcabc):
    """
    Run gcMC based on given Ca/Na concentration cell.
    Initial structure is the lowest Ewald energy structure.
    """
    def __init__(self,
                 machine,
                 dmus,
                 savepath,
                 savename,
                 ce_file_path,
                 ensemble_file_path,
                 ca_amt,
                 na_amt,
                 steps,
                 temperature,
                 thin_by,
                 discard
                 ):
        """
        Args:
            machine: Machine that will run jobs.
            dmus: list(tuple). Chemical potentials of Na, Ca that will be used.
                  First one of tuple is Na chemical potential.
            savepath: Directory want to save hdf5 file.
            savename: Name of hdf5 file to save.
            ce_file_path: Cluster Expansion Object path.
            ensemble_file_path: Ensemble Object path.
            ca_amt: Initial Ca amount. Unit is per V2(PO4)3 formula.
            na_amt: Initial Na amount. Unit is per V2(PO4)3 formula.
            steps: MC steps to run. Default to 5000000
            temperature: T want to use.
            thin_by: Thin
            discard: Discard
        """
        super().__init__(machine, dmus, savepath, savename, ce_file_path, ensemble_file_path)
        self.ca_amt = ca_amt
        self.na_amt = na_amt
        self.steps = steps
        self.temperature = temperature
        self.thin_by = thin_by
        self.discard = discard

    def get_intermediate_structure(self):
        """
        For given supercell matrix, make intermediate supercell with the shortest elongated
        direction. This will be used to generate intermediate size of ordered structure.
        """
        interm_matrix, residue_matrix = divide_matrix(self.supercell_matrix)
        interm_cell = deepcopy(self.expansion.structure)
        interm_cell.make_supercell(interm_matrix)

        return interm_cell, residue_matrix

    def initialized_structure(self, verbose=True) -> np.ndarray:
        """
        Initialize structure based on Ewald ordering.
        """
        start = time.time()

        interm_cell, residue_matrix = self.get_intermediate_structure()

        for i, j in enumerate(interm_cell):
            if j.species_string == "Na+:0.333, Ca2+:0.333":
                interm_cell.replace(i, 'Na')
            elif j.species_string == "V3+:0.333, V4+:0.333, V5+:0.333":
                interm_cell.replace(i, 'V')

        poscar_generator = PoscarGen(basestructure=interm_cell)
        groups = poscar_generator.get_ordered_structure(self.ca_amt, self.na_amt)

        if 'Ni3+' in groups[0][0].composition.as_dict().keys():
            groups[0][0].replace_species({'Ni3+': 'V3+'})
        if 'Mn4+' in groups[0][0].composition.as_dict().keys():
            groups[0][0].replace_species({'Mn4+': 'V4+'})
        if 'Cr5+' in groups[0][0].composition.as_dict().keys():
            groups[0][0].replace_species({'Cr5+': 'V5+'})

        final_supercell = groups[0][0]
        final_supercell.make_supercell(residue_matrix)
        occupancy = self.ensemble.processor.occupancy_from_structure(final_supercell)
        end = time.time()

        if verbose:
            print(f"{end - start}s for initialization.\n")
        return occupancy

    def run(self):

        init_occu = self.ensemble.processor.occupancy_from_structure()

        for dmu in self.dmus:
            chempo = {'Na+': dmu[1], 'Ca2+': dmu[0], 'Vacancy': 0, 'V3+': 0, 'V4+': 0, 'V5+': 0}
            running_ensemble = self.ensemble
            running_ensemble.chemical_potentials = chempo
            sampler = Sampler.from_ensemble(running_ensemble,
                                            step_type="tableflip",
                                            temperature=self.temperature,
                                            optimize_basis=False,
                                            flip_table=self.flip_table)
            sampler.run(nsteps=self.steps,
                        init_occu=init_occu,
                        thin_by=self.thin_by,
                        progress=False)

            # Update flip reactions to sampler metadata.
            bits = sampler.mckernels[0].mcusher.bits
            flip_table = sampler.mckernels[0].mcusher.flip_table
            flip_reaction = [flip_vec_to_reaction(u, bits) for u in flip_table]
            sampler.samples.metadata['flip_reaction'] = flip_reaction

            # Saving. TODO: Use flush to backend and do not call sampler everytime.
            filename = "{}_{}_cn_sgmc.mson".format(dmu[0], dmu[1])
            filepath = self.savepath.replace("test_samples.mson", filename)
            sampler.samples.to_hdf5(filepath)
            print(f"Sampling information: {sampler.samples.metadata}\n")
            print("Ca chempo: {}, Na chempo: {} is done. Check {}\n".
                  format(dmu[0], dmu[1], filepath))

        return

    def sanity_check(self):

        return


def main(machine,
         ca_dmu=None,
         na_dmu=None,
         savepath=None,
         ca_amt=0.5,
         na_amt=1.0,
         steps=5000000,
         temperature=300,
         thin_by=10,
         discard=0
         ):

    ca_dmu_float = []
    na_dmu_float = []
    dmus = []

    # Handling input string list to float list
    for i in ca_dmu:
        ca_dmu_float.append(float(''.join(i)))
    for i in na_dmu:
        na_dmu_float.append(float(''.join(i)))

    assert len(na_dmu_float) == len(ca_dmu_float)

    for i, j in enumerate(ca_dmu_float):
        dmus.append((float(j), float(na_dmu_float[i])))

    runner = gcmc_basic(machine=machine,
                        dmus=dmus,
                        savepath=savepath,
                        savename=None,
                        ce_file_path=None,
                        ensemble_file_path=None,
                        ca_amt=ca_amt,
                        na_amt=na_amt,
                        steps=steps,
                        temperature=temperature,
                        thin_by=thin_by,
                        discard=discard
                        )
    runner.run()

    return


def test():

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ca_amt', type=float, required=False, default=0.5,
                        help="Amount of Ca in initial structure.")
    parser.add_argument('--na_amt', type=float, required=False, default=1.0,
                        help="Amount of Na in initial structure.")
    parser.add_argument('--ca_dmu', nargs="+", type=list, required=False, default=None,
                        help="List of Ca chemical potentials.")
    parser.add_argument('--na_dmu', nargs="+", type=list, required=False, default=None,
                        help="List of Na chemical potentials.")
    parser.add_argument('--path', type=str, required=False, default=None,
                        help="Path want to save")
    args = parser.parse_args()
    main(machine="savio",
         ca_dmu=args.ca_dmu,
         na_dmu=args.na_dmu,
         savepath=args.path,
         ca_amt=args.ca_amt,
         na_amt=args.na_amt)
