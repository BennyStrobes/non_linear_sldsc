#!/bin/bash
#SBATCH -t 0-48:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)
#SBATCH -c 20                               # Request one core



trait_name="$1"
preprocessed_data_for_non_linear_sldsc_dir="$2"
non_linear_sldsc_results_dir="$3"
model_type="$4"
samp_size="$5"
training_chromosomes="$6"
evaluation_chromosomes="$7"
ld_type="$8"
learn_intercept="$9"
learning_rate="${10}"



if false; then
module load gcc/6.2.0
module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate
fi

conda deactivate
module purge
module load gcc/9.2.0
module load python/3.9.14
module load cuda/12.1
source /n/groups/price/ben/environments/tf_gpu/bin/activate

python run_non_linear_sldsc_marginal_updates.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $model_type $samp_size $training_chromosomes $evaluation_chromosomes $ld_type $learn_intercept $learning_rate