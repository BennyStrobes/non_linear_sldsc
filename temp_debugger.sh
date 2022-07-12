#!/bin/bash
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=10G                         # Memory total in MiB (for all cores)



trait_name="$1"
preprocessed_data_for_non_linear_sldsc_dir="$2"
non_linear_sldsc_results_dir="$3"
samp_size="$4"




module load gcc/6.2.0
module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate

if false; then
python3 temp_debugger.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $samp_size "even"
fi

if false; then
python3 temp_debugger_old.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $samp_size "even"
fi

if false; then
python3 temp_debugger.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $samp_size "odd"
fi

if false; then
python3 temp_debugger_old.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $samp_size "odd"
fi

if false; then
python3 temp_debugger_iterative.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $samp_size "even"
fi
if false; then
python3 temp_debugger_iterative.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $samp_size "odd"
fi

if false; then
source ~/.bash_profile
module load R/3.5.1
Rscript visualize_temp_debugger.R $trait_name $non_linear_sldsc_results_dir
fi

echo "heloo"
python3 temp_debugger.py $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir $samp_size "chrom_5"
