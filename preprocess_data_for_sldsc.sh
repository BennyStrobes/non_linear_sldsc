#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)

 

ukbb_preprocessed_for_genome_wide_susie_dir="$1"
ldsc_baseline_ld_hg38_annotation_dir="$2"
preprocessed_data_for_non_linear_sldsc_dir="$3"
preprocessed_data_for_sldsc_dir="$4"
ref_1kg_genotype_dir="$5"


if false; then
for chrom_num in {1..22}; do 
	sbatch preprocess_data_for_sldsc_per_chrom.sh $ukbb_preprocessed_for_genome_wide_susie_dir $ldsc_baseline_ld_hg38_annotation_dir $preprocessed_data_for_non_linear_sldsc_dir $chrom_num $preprocessed_data_for_sldsc_dir $ref_1kg_genotype_dir
done
fi
