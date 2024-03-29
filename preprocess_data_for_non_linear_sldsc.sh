#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)




ukbb_preprocessed_for_genome_wide_susie_dir="$1"
ldsc_baseline_ld_hg19_annotation_dir="$2"
preprocessed_data_for_non_linear_sldsc_dir="$3"
trait_name="$4"

if false; then
for chrom_num in {1..22}; do 
	sbatch preprocess_data_for_non_linear_sldsc_per_chrom.sh $ukbb_preprocessed_for_genome_wide_susie_dir $ldsc_baseline_ld_hg19_annotation_dir $preprocessed_data_for_non_linear_sldsc_dir $chrom_num $trait_name
done
fi


python3 calculate_total_number_of_snps.py $preprocessed_data_for_non_linear_sldsc_dir $trait_name
