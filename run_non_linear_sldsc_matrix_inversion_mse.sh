#!/bin/bash
#SBATCH -t 0-40:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=40G                         # Memory total in MiB (for all cores)



trait_name="$1"
preprocessed_data_for_non_linear_sldsc_dir="$2"
samp_size="$3"
diagonal_padding="$4"
non_linear_sldsc_results_dir="$5"
model_type="$6"

module load gcc/6.2.0
module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate


python3 run_non_linear_sldsc_matrix_inversion_mse.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $samp_size $diagonal_padding $non_linear_sldsc_results_dir $model_type
