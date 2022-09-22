#!/bin/bash
#SBATCH --job-name=pytest
#SBATCH --account=fc_ceder
#SBATCH --partition=savio
#SBATCH --qos=savio_debug
#SBATCH --output=log.o
#SBATCH --error=log.e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=20

# Load mpi modules.
module load intel/2016.4.072
module load mkl/2016.4.072
module load openmpi/2.0.2-intel
module load fftw/3.3.7

# If you run python job.
module load python/3.9.12
source activate cn-sgmc
python /global/scratch/users/yychoi94/CaNaVP/gcmc_script/basic_script.py --ca_amt 0.5 --na_amt 0.5 --ca_dmu -2 -3 -4 --na_dmu -3 -4 -5 > py.out
