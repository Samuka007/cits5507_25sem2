#!/bin/bash
#SBATCH --job-name=mpi_benchmark
#SBATCH --nodes=2
#SBATCH --ntasks=4 # total MPI tasks
#SBATCH --time=01:00:00
#SBATCH --mem=1024G
#SBATCH --partition=cits3402
#SBATCH -o logs/proc_2n_4p_96th_44w_7_%j.out
#SBATCH -e logs/proc_2n_4p_96th_44w_7_%j.err
#SBATCH --mail-user=24754678@student.uwa.edu.au
#SBATCH --mail-type=BEGIN,END
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
# Load compiler + MPI
module load gcc/14.3
module load openmpi/5.0.5
echo "Job started at: $(date)"
make PROCESS=$SLURM_NPROCS RANK=440000 KERNEL=7 benchmark
echo "Job finished at: $(date)"
