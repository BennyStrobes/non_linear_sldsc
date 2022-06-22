#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)




ukbb_preprocessed_for_genome_wide_susie_dir="$1"
ldsc_baseline_ld_hg38_annotation_dir="$2"
preprocessed_data_for_non_linear_sldsc_dir="$3"
chrom_num="$4"
trait_name="$5"

module load gcc/6.2.0
module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate


python3 preprocess_data_for_non_linear_sldsc_per_chrom.py $ukbb_preprocessed_for_genome_wide_susie_dir $ldsc_baseline_ld_hg38_annotation_dir $preprocessed_data_for_non_linear_sldsc_dir $chrom_num $trait_name
