#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)


trait_name="$1"
preprocessed_data_for_sldsc_dir="$2"
samp_size="$3"
non_linear_sldsc_results_dir="$4"
ukbb_sumstats_hg38_dir="$5"
ldsc_code_dir="$6"


source /n/groups/price/ben/environments/sldsc/bin/activate
module load python/2.7.12
if false; then
python organize_sumstats_for_sldsc.py $trait_name $samp_size $preprocessed_data_for_sldsc_dir $ukbb_sumstats_hg38_dir
fi

# All chromosomes
if false; then
python ${ldsc_code_dir}ldsc.py --h2 ${preprocessed_data_for_sldsc_dir}${trait_name}"_all_chrom.sumstats" --ref-ld-chr ${preprocessed_data_for_sldsc_dir}"baselineLD." --w-ld-chr ${preprocessed_data_for_sldsc_dir}"weights.hm3_noMHC." --overlap-annot --print-coefficients --frqfile-chr ${preprocessed_data_for_sldsc_dir}"1000G.EUR.hg38." --out ${non_linear_sldsc_results_dir}${trait_name}"_sldsc_source_code_all_chrom"

python ${ldsc_code_dir}ldsc.py --h2 ${preprocessed_data_for_sldsc_dir}${trait_name}"_even_chrom.sumstats" --ref-ld-chr ${preprocessed_data_for_sldsc_dir}"baselineLD." --w-ld-chr ${preprocessed_data_for_sldsc_dir}"weights.hm3_noMHC." --overlap-annot --print-coefficients --frqfile-chr ${preprocessed_data_for_sldsc_dir}"1000G.EUR.hg38." --out ${non_linear_sldsc_results_dir}${trait_name}"_sldsc_source_code_even_chrom"

python ${ldsc_code_dir}ldsc.py --h2 ${preprocessed_data_for_sldsc_dir}${trait_name}"_odd_chrom.sumstats" --ref-ld-chr ${preprocessed_data_for_sldsc_dir}"baselineLD." --w-ld-chr ${preprocessed_data_for_sldsc_dir}"weights.hm3_noMHC." --overlap-annot --print-coefficients --frqfile-chr ${preprocessed_data_for_sldsc_dir}"1000G.EUR.hg38." --out ${non_linear_sldsc_results_dir}${trait_name}"_sldsc_source_code_odd_chrom"

fi

python ${ldsc_code_dir}ldsc.py --h2 ${preprocessed_data_for_sldsc_dir}${trait_name}"_all_chrom.sumstats" --ref-ld-chr ${preprocessed_data_for_sldsc_dir}"baselineLD." --w-ld-chr ${preprocessed_data_for_sldsc_dir}"weights.hm3_noMHC." --overlap-annot --no-intercept --print-coefficients --frqfile-chr ${preprocessed_data_for_sldsc_dir}"1000G.EUR.hg38." --out ${non_linear_sldsc_results_dir}${trait_name}"_sldsc_source_code_all_chrom_no_intercept"

python ${ldsc_code_dir}ldsc.py --h2 ${preprocessed_data_for_sldsc_dir}${trait_name}"_even_chrom.sumstats" --ref-ld-chr ${preprocessed_data_for_sldsc_dir}"baselineLD." --w-ld-chr ${preprocessed_data_for_sldsc_dir}"weights.hm3_noMHC." --overlap-annot --no-intercept --print-coefficients --frqfile-chr ${preprocessed_data_for_sldsc_dir}"1000G.EUR.hg38." --out ${non_linear_sldsc_results_dir}${trait_name}"_sldsc_source_code_even_chrom_no_intercept"

python ${ldsc_code_dir}ldsc.py --h2 ${preprocessed_data_for_sldsc_dir}${trait_name}"_odd_chrom.sumstats" --ref-ld-chr ${preprocessed_data_for_sldsc_dir}"baselineLD." --w-ld-chr ${preprocessed_data_for_sldsc_dir}"weights.hm3_noMHC." --overlap-annot --no-intercept --print-coefficients --frqfile-chr ${preprocessed_data_for_sldsc_dir}"1000G.EUR.hg38." --out ${non_linear_sldsc_results_dir}${trait_name}"_sldsc_source_code_odd_chrom_no_intercept"














###############
# OLD
###############

if false; then
formatted_sumstats_dir="/n/groups/price/ldsc/sumstats_formatted_2021/"

# annotation dir
annotation_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3_hg38/baselineLD_v2.2/"

# Weights dir
weights_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3_hg38/weights/"

# Frq dir
frq_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3_hg38/plink_files/"

python ${ldsc_code_dir}ldsc.py --h2 ${formatted_sumstats_dir}"UKB_460K."${trait_name}".sumstats" --ref-ld-chr ${annotation_dir}"baselineLD." --w-ld-chr ${weights_dir}"weights.hm3_noMHC." --overlap-annot --print-coefficients --frqfile-chr ${frq_dir}"1000G.EUR.hg38." --out ${preprocessed_data_for_sldsc_dir}${trait_name}"_sldsc"
fi