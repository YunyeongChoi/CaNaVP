#!/bin/bash
#SBATCH --job-name=pytest
#SBATCH --account=co_condoceder
#SBATCH --partition=savio4_htc
#SBATCH --qos=condoceder_htc4_normal
#SBATCH --output=log.o
#SBATCH --error=log.e
#SBATCH --time=03:00:00
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1

# Load mpi modules.
module load intel/2016.4.072
module load mkl/2016.4.072
module load openmpi/2.0.2-intel
module load fftw/3.3.7

# If you run python job.
module load python/3.9.12
source activate cn-sgmc
python /global/scratch/users/yychoi94/CaNaVP/gcmc/launcher/basic_launcher.py --ca_amt 0.5 --na_amt 1.0 --ca_dmu -8.5 -9.0 -9.5 --na_dmu -4.2 -4.5 -4.75 --path "/global/scratch/users/yychoi94/CaNaVP_gcMC/test"> py.out
