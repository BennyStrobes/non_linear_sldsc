#!/bin/bash
#SBATCH -t 0-40:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)



trait_name="$1"
preprocessed_data_for_non_linear_sldsc_dir="$2"
non_linear_sldsc_results_dir="$3"
model_type="$4"
samp_size="$5"




module load gcc/6.2.0
module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate



python3 run_non_linear_sldsc_marginal_updates_mse_unweighted.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $model_type $samp_size