


##################
# Input data
##################

# Directory containing summary statistics
ukbb_sumstats_hg19_dir="/n/groups/price/UKBiobank/sumstats/bolt_337K_unrelStringentBrit_MAF0.001_v3/"

#Directory that contains necessary liftover information.
##Specifically, it must contain:
#####1. 'liftOver'   --> the executable
#####2. 'hg19ToHg38.over.chain.gz'   --> for converting from hg19 to hg38
#####2. 'hg38ToHg19.over.chain.gz'   --> for converting from hg38 to hg19
liftover_directory="/n/groups/price/ben/tools/liftOver_x86/"

# GTEx genotype dir
gtex_genotype_dir="/n/groups/price/ben/eqtl_informed_prs/gtex_v8_meta_analysis_eqtl_calling/pseudotissue_genotype/"

# Genotype data from 1KG
ref_1kg_genotype_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3_hg38/plink_files/"

# LDSC baselineLD annotations (hg38)
ldsc_baseline_ld_hg38_annotation_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3_hg38/baselineLD_v2.2/"


##################
# Output data
##################
# old_Output root directory
old_output_root="/n/groups/price/ben/causal_eqtl_gwas/gtex_v8_causal_eqtl_gwas_38_tissues/"

# Directory containing hg38 ukbb summary stats
ukbb_sumstats_hg38_dir=$old_output_root"ukbb_sumstats_hg38/"

# Directory containing UKBB sumstats for genome-wide susie
ukbb_preprocessed_for_genome_wide_susie_dir=$old_output_root"ukbb_preprocessed_for_genome_wide_susie/"


# Output root
output_root="/n/groups/price/ben/non_linear_sldsc/"

# Directory containing preprocessed data for non-linear sldsc analysis
preprocessed_data_for_non_linear_sldsc_dir=$output_root"preprocessed_data_for_non_linear_sldsc/"

# Directory containing non-linear sldsc analysis
non_linear_sldsc_results_dir=$output_root"non_linear_sldsc_results/"


##################
# Analysis
##################

########################################
# Liftover UKBB summary statistics to hg38
# NOTE: THIS WAS PREVIOUSLY RUN. so just using for convenience
########################################
if false; then
sbatch liftover_ukbb_summary_statistics_from_hg19_to_hg38.sh $liftover_directory $ukbb_sumstats_hg19_dir $ukbb_sumstats_hg38_dir
fi


########################################
# # Preprocess data for UKBB genome-wide Susie Analysis
# NOTE: THIS WAS PREVIOUSLY RUN. so just using for convenience
########################################
if false; then
sh preprocess_data_for_genome_wide_ukbb_susie_analysis.sh $ukbb_sumstats_hg38_dir $gtex_genotype_dir $ref_1kg_genotype_dir $ukbb_preprocessed_for_genome_wide_susie_dir
fi



########################################
# # Preprocess data for non-linear S-LDSC analysis
########################################
trait_name="blood_WHITE_COUNT"
if false; then
sh preprocess_data_for_non_linear_sldsc.sh $ukbb_preprocessed_for_genome_wide_susie_dir $ldsc_baseline_ld_hg38_annotation_dir $preprocessed_data_for_non_linear_sldsc_dir $trait_name
fi

########################################
# # Run non-linear S-LDSC analysis
########################################
trait_name="blood_WHITE_COUNT"

sh run_non_linear_sldsc.sh $trait_name $preprocessed_data_for_non_linear_sldsc_dir $non_linear_sldsc_results_dir



