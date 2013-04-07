#!/bin/bash
#SBATCH --job-name=SCHULD6_ASSIGNMENT4
#SBATCH --time=01:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --output=/gpfs/sb/home/PCP3/PCP3schl/assignment4/a4_output.%J
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schuld6@rpi.edu


srun --nodes=1 --ntasks=64 --overcommit /gpfs/sb/home/PCP3/PCP3schl/assignment4/bin/matrix_multiply 8192 1
srun --nodes=1 --ntasks=32 --overcommit /gpfs/sb/home/PCP3/PCP3schl/assignment4/bin/matrix_multiply 8192 2
srun --nodes=1 --ntasks=16 --overcommit /gpfs/sb/home/PCP3/PCP3schl/assignment4/bin/matrix_multiply 8192 4
srun --nodes=1 --ntasks=8 --overcommit /gpfs/sb/home/PCP3/PCP3schl/assignment4/bin/matrix_multiply 8192 8
srun --nodes=1 --ntasks=4 --overcommit /gpfs/sb/home/PCP3/PCP3schl/assignment4/bin/matrix_multiply 8192 16
srun --nodes=1 --ntasks=2 --overcommit /gpfs/sb/home/PCP3/PCP3schl/assignment4/bin/matrix_multiply 8192 32
srun --nodes=1 --ntasks=1 --overcommit /gpfs/sb/home/PCP3/PCP3schl/assignment4/bin/matrix_multiply 8192 64