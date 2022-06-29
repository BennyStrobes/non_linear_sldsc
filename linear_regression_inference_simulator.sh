#!/bin/bash
#SBATCH -c 20                               # Request one core
#SBATCH -t 0-70:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=60G                         # Memory total in MiB (for all cores)





module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate


python3 linear_regression_inference_simulator.py