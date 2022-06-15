import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas
import os
import pdb
import tensorflow as tf
import gzip



# Convert gwas summary statistics to *STANDARDIZED* effect sizes
# Following SuSiE code found in these two places:
########1. https://github.com/stephenslab/susieR/blob/master/R/susie_rss.R  (LINES 277-279)
########2. https://github.com/stephenslab/susieR/blob/master/R/susie_ss.R (LINES 148-156 AND 203-205)
def convert_to_standardized_summary_statistics(gwas_beta_raw, gwas_beta_se_raw, gwas_sample_size, R, sigma2=1.0):
	gwas_z_raw = gwas_beta_raw/gwas_beta_se_raw

	XtX = (gwas_sample_size-1)*R
	Xty = np.sqrt(gwas_sample_size-1)*gwas_z_raw
	var_y = 1

	dXtX = np.diag(XtX)
	csd = np.sqrt(dXtX/(gwas_sample_size-1))
	csd[csd == 0] = 1

	XtX = (np.transpose((1/csd) * XtX) / csd)
	Xty = Xty / csd

	dXtX2 = np.diag(XtX)

	beta_scaled = (1/dXtX2)*Xty
	beta_se_scaled = np.sqrt(sigma2/dXtX2)

	return beta_scaled, beta_se_scaled

def create_variant_to_genomic_annotation_mapping(chrom_annotation_file):
	f = gzip.open(chrom_annotation_file)
	head_count = 0
	dicti = {}
	for line in f:
		line = line.decode("utf-8").rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			anno_names = np.asarray(data[4:])
			continue
		short_variant_id = 'chr' + data[0] + '_' + data[1]
		annotations = np.asarray(data[4:]).astype(float)
		#if short_variant_id in dicti:
			#print('assumption eroror')
			#pdb.set_trace()
		dicti[short_variant_id] = annotations
	f.close()
	return anno_names, dicti

def extract_variant_indices_that_we_have_genomic_annotations_for(variant_ids, variant_to_genomic_annotations):
	indices = []
	for ii, variant_id in enumerate(variant_ids):
		variant_info = variant_id.split('_')
		short_variant_id = variant_info[0] + '_' + variant_info[1]
		if short_variant_id in variant_to_genomic_annotations:
			indices.append(ii)
	return np.asarray(indices)


def generate_rss_likelihood_relevent_statistics(beta_scaled, beta_se_scaled, trait_sample_size, ld):
	s_squared_vec = np.square(beta_se_scaled) + (np.square(beta_scaled)/trait_sample_size)
	s_vec = np.sqrt(s_squared_vec)
	S_mat = np.diag(s_vec)
	S_inv_mat = np.diag(1.0/s_vec)
	S_inv_2_mat = np.diag(1.0/np.square(s_vec))
	# Compute (S^-1)R(S^-1) taking advantage of fact that S^-1 is a diagonal matrix
	D_mat = np.multiply(np.multiply(np.diag(S_inv_mat)[:, None], ld), np.diag(S_inv_mat))
	# Compute (S)R(S^-1) taking advantage of fact that S and S^-1 is a diagonal matrix
	srs_inv_mat = np.multiply(np.multiply(np.diag(S_mat)[:, None], ld), np.diag(S_inv_mat))
	s_inv_2_diag = np.diag(S_inv_2_mat)
	D_diag = np.diag(D_mat)

	return srs_inv_mat, s_inv_2_diag, D_diag


def extract_middle_variant_indices(variant_ids, window_start, window_end):
	middle_start = window_start + 1000000
	middle_end = window_end - 1000000

	# Quick error checking
	if middle_end - middle_start - 1000000 != 0:
		print('assumptino erroror')
		pdb.set_trace()


	# Now get indices that are greater than or equal to middle_start and less than middle_end
	middle_indices = []
	for index, variant_id in enumerate(variant_ids):
		variant_position = float(variant_id.split('_')[1])

		if variant_position >= middle_start and variant_position < middle_end:
			middle_indices.append(index)
	return np.asarray(middle_indices)




ukbb_preprocessed_for_genome_wide_susie_dir = sys.argv[1]  # Input dir
ldsc_baseline_ld_hg38_annotation_dir = sys.argv[2]  # Input dir (genomic annotations)
output_dir = sys.argv[3]
chrom_num = sys.argv[4]
trait_name = sys.argv[5]


# First, load in genomic annotations of all variants on this chromosome
chrom_annotation_file = ldsc_baseline_ld_hg38_annotation_dir + 'baselineLD.' + chrom_num + '.annot.gz'
annotation_names, variant_to_genomic_annotations = create_variant_to_genomic_annotation_mapping(chrom_annotation_file)

# Open file handle for output file
output_window_file = output_dir + trait_name + '_genome_wide_susie_windows_and_non_linear_sldsc_processed_data_chrom_' + chrom_num + '.txt'
t = open(output_window_file,'w')
t.write('window_name' + '\tvariant_id\tsrs_inv\ts_inv_2_diag\tD_diag\tbeta\tgenomic_annotation\tmiddle_variant_indices\n')

# Open file handle for input file
input_window_file = ukbb_preprocessed_for_genome_wide_susie_dir + 'genome_wide_susie_windows_and_processed_data_chrom_' + chrom_num + '.txt'
f = open(input_window_file)


# Stream input file
head_count = 0
counter = 0
for line in f:
	line = line.rstrip()
	data = line.split('\t')
	if head_count == 0:
		head_count = head_count + 1
		continue
	counter = counter + 1
	print(counter)
	# Extract relevent fields from line
	window_name = data[0] + ':' + data[1] + ':' + data[2]
	window_start = int(data[1])
	window_end = int(data[2])
	beta_file = data[5]
	beta_std_err_file = data[6]
	variant_file = data[7]
	study_file = data[8]
	ref_genotype_file = data[9]
	sample_size_file = data[10]

	##############################
	# Load in data
	##############################
	# Variant ids
	variant_ids = np.loadtxt(variant_file,dtype=str)
	variant_indices = extract_variant_indices_that_we_have_genomic_annotations_for(variant_ids, variant_to_genomic_annotations)
	variant_ids = variant_ids[variant_indices]
	
	# study file
	all_studies = np.loadtxt(study_file,dtype=str)
	# QUick error check
	if len(np.where(all_studies==trait_name)[0]) != 1:
		print('assumption error')
		pdb.set_trace()
	trait_index = np.where(all_studies==trait_name)[0][0]

	# extract trait sample size
	all_sample_sizes = np.loadtxt(sample_size_file)
	trait_sample_size = all_sample_sizes[trait_index]

	# Load in LD matrix
	geno = np.loadtxt(ref_genotype_file)
	geno = geno[:, variant_indices]
	ld = np.corrcoef(np.transpose(geno))

	# Load in gwas summary statistics
	# betas
	all_betas = np.loadtxt(beta_file)
	betas = all_betas[trait_index,:]
	betas = betas[variant_indices]
	# beta std errors
	all_beta_std_err = np.loadtxt(beta_std_err_file)
	beta_std_errs = all_beta_std_err[trait_index,:]
	beta_std_errs = beta_std_errs[variant_indices]

	# Genomic annotations
	genomic_anno_mat = []
	for variant_id in variant_ids:
		variant_info = variant_id.split('_')
		short_variant_id = variant_info[0] + '_' + variant_info[1]
		genomic_anno_mat.append(variant_to_genomic_annotations[short_variant_id])
	genomic_anno_mat = np.asarray(genomic_anno_mat)

	##############################
	# Prepare data
	##############################
	# Scale summary statistics so it was though they were run with standardized genotype
	beta_scaled, beta_se_scaled = convert_to_standardized_summary_statistics(betas, beta_std_errs, trait_sample_size, ld)

	# Extract data objects use for running rss likelihood
	srs_inv, s_inv_2_diag, D_diag = generate_rss_likelihood_relevent_statistics(beta_scaled, beta_se_scaled, trait_sample_size, ld)
	

	# Extract indices of middle variants
	middle_variant_indices = extract_middle_variant_indices(variant_ids, window_start, window_end)

	if len(middle_variant_indices) < 10:
		continue
	##############################
	# Save data
	# Things to save:
	### 1. variant IDS
	### 2. srs_inv mat
	### 3. s_inv_2_diag mat
	### 4. D_diag mat
	### 5. beta_scaled
	### 6. genomic_anno_mat
	##############################
	window_output_root = output_dir + trait_name + '_' + window_name + '_'

	# Variant Ids
	variant_id_output_file = window_output_root + 'variant_ids.npy'
	np.save(variant_id_output_file, variant_ids)

	# srs_inv mat
	srs_inv_output_file = window_output_root + 'srs_inv.npy'
	np.save(srs_inv_output_file, srs_inv)

	# s_inv_2_diag mat
	s_inv_2_diag_output_file = window_output_root + 's_inv_2_diag.npy'
	np.save(s_inv_2_diag_output_file, s_inv_2_diag)

	# D_diag mat
	D_diag_output_file = window_output_root + 'D_diag.npy'
	np.save(D_diag_output_file, D_diag)

	# beta-scaled
	beta_output_file = window_output_root + 'beta.npy'
	np.save(beta_output_file, beta_scaled)

	# genomic annotations
	genomic_anno_output_file = window_output_root + 'genomic_anno.npy'
	np.save(genomic_anno_output_file, genomic_anno_mat)

	# middle variant indices
	middle_window_output_file = window_output_root + 'middle_window_variants.npy'
	np.save(middle_window_output_file, middle_variant_indices)

	# Print file names to global output file
	t.write(window_name + '\t' + variant_id_output_file + '\t' + srs_inv_output_file + '\t' + s_inv_2_diag_output_file + '\t' + D_diag_output_file + '\t' + beta_output_file + '\t' + genomic_anno_output_file + '\t' + middle_window_output_file + '\n')

f.close()
t.close()










