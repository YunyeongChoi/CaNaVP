#!/bin/bash
#SBATCH --error=log.e
#SBATCH --out=log.o
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --job-name=calc_test-0.5_1.0
#SBATCH --account=co_condoceder
#SBATCH --partition=savio3
#SBATCH --qos=savio_lowprio

mkdir U; cd U;
IsConv=`grep 'required accuracy' OUTCAR`;
if [ -z "${IsConv}" ]; then
    if [ -s "CONTCAR" ]; then cp CONTCAR POSCAR; fi;
    if [ ! -s "POSCAR" ]; then
        cp ../{KPOINTS,POTCAR,POSCAR} .;
    fi
        cp ../INCAR .;
mpirun -n 128 /global/home/users/yychoi94/bin/vasp.5.4.4_vtst178_with_DnoAugXCMeta/vasp_std > vasp.out
fi