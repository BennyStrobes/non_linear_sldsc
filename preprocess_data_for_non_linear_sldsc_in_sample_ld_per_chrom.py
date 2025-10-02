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

	return beta_scaled, beta_se_scaled, XtX, Xty

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

	return srs_inv_mat, s_inv_2_diag, D_diag, D_mat


def extract_middle_variant_indices(variant_ids, window_start, window_end, regression_snps):
	middle_start = window_start + 1000000
	middle_end = window_end - 1000000

	# Quick error checking
	if middle_end - middle_start - 1000000 != 0:
		print('assumptino erroror')
		pdb.set_trace()


	# Now get indices that are greater than or equal to middle_start and less than middle_end
	middle_indices = []
	regression_indices = []
	for index, variant_id in enumerate(variant_ids):
		variant_position = float(variant_id.split('_')[1])
		short_variant_id = '_'.join(variant_id.split('_')[:2])
		if short_variant_id in regression_snps:
			regression_indices.append(index)
		if variant_position >= middle_start and variant_position < middle_end:
			middle_indices.append(index)
	return np.asarray(middle_indices), np.asarray(regression_indices)


def debugger(XtX_special, Xty):
	expected_tau = 1000000.0
	num_snps = len(Xty)
	print(num_snps)
	for itera in range(100):
		S = np.linalg.inv( np.diag(np.ones(num_snps)*expected_tau) + XtX_special )
		mu = np.dot(S, Xty)

		tau_a = num_snps/2.0
		tau_b = 0.5*np.sum(np.square(mu) + np.diag(S))
		expected_tau = tau_a/tau_b
		print(expected_tau)

def debug_extract_alt_betas(betas_orig, beta_std_errs_orig, ld, variant_ids, chrom_num):
	dicti = {}
	f = open('/n/groups/price/ben/non_linear_sldsc/non_bolt_lmm_hg38_sumstats/blood_WHITE_COUNT_hg38_liftover.bgen.stats')
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			print(data)
			continue
		if data[1] != chrom_num:
			continue
		variant_id_1 = 'chr' + data[1] + '_' + data[2] + '_' + data[3] + '_' + data[4]
		variant_id_2 = 'chr' + data[1] + '_' + data[2] + '_' + data[4] + '_' + data[3]
		beta = float(data[7])
		se = float(data[8])
		new_trait_samp_size = int(data[10])
		dicti[variant_id_1] = (beta,se)
		dicti[variant_id_2] = (beta,se)
	f.close()

	valid_indices = []
	new_betas = []
	new_beta_se = []
	for variant_iter, variant_id in enumerate(variant_ids):
		if variant_id not in dicti:
			continue
		valid_indices.append(variant_iter)
		unsigned_beta = np.abs(dicti[variant_id][0])
		unsigned_beta_se = dicti[variant_id][1]
		if betas_orig[variant_iter] < 0.0:
			signed_beta = unsigned_beta*-1.0
		else:
			signed_beta = unsigned_beta*1.0
		new_betas.append(signed_beta)
		new_beta_se.append(unsigned_beta_se)


	# NEW DATA
	new_betas = np.asarray(new_betas)
	new_beta_se = np.asarray(new_beta_se)
	valid_indices = np.asarray(valid_indices)
	new_variant_ids = variant_ids[valid_indices]
	new_ld = ld[valid_indices,:][:,valid_indices]


	new_beta_scaled, new_beta_se_scaled, new_XtX_special, new_Xty = convert_to_standardized_summary_statistics(new_betas, new_beta_se, new_trait_samp_size, new_ld)

	new_srs_inv, new_s_inv_2_diag, new_D_diag, new_D_mat = generate_rss_likelihood_relevent_statistics(new_beta_scaled, new_beta_se_scaled, new_trait_samp_size, new_ld)

	debugger(new_D_mat, (new_s_inv_2_diag*new_beta_scaled))

def extract_hm3_snps(hm3_snp_file):
	f = gzip.open(hm3_snp_file)
	head_count = 0
	hm3_snps = {}
	for line in f:
		line = line.decode("utf-8").rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		short_variant_id = 'chr' + data[0] + '_' + data[2]
		hm3_snps[short_variant_id] = 1
	f.close()
	return hm3_snps

def extract_dictionary_list_of_all_regression_snps(input_window_file, hm3_snps):
	regression_snps = {}
	head_count = 0
	f = open(input_window_file)
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		# Extract relevent fields from line
		window_name = data[0] + ':' + data[1] + ':' + data[2]
		window_start = int(data[1])
		window_end = int(data[2])
		middle_start = window_start + 1000000
		middle_end = window_end - 1000000
		variant_file = data[7]

		beta_file = data[5]
		beta_std_err_file = data[6]
		all_betas = np.loadtxt(beta_file)
		all_beta_std_err = np.loadtxt(beta_std_err_file)
		# Ignore windows with nans
		z = all_betas/all_beta_std_err
		if np.sum(np.isnan(z)) > 0:
			print('reg skkip')
			continue
		if np.sum(np.isnan(all_betas)) > 0:
			print('reg skip 2')
			continue

		# Load in data
		# Variant ids
		variant_ids = np.loadtxt(variant_file,dtype=str)
		for variant_id in variant_ids:
			variant_position = float(variant_id.split('_')[1])

			if variant_position >= middle_start and variant_position < middle_end:
				short_variant_id = '_'.join(variant_id.split('_')[:2])
				if short_variant_id in hm3_snps:
					regression_snps[short_variant_id] = 1
	f.close()
	return regression_snps


def get_window_sequence_from_samtools_output(temp_fasta_output_file):
	f = open(temp_fasta_output_file)
	arr = []
	for line in f:
		line = line.rstrip()
		if line.startswith('>chr'):
			continue
		for nucleotide in line:
			arr.append(nucleotide)
	f.close()

	return np.asarray(arr)

def error_check_to_make_sure_variant_id_reference_matches_fasta_reference(window_sequence, window_start, variant_ids):
	refs = []
	alts = []
	for ii,variant_id in enumerate(variant_ids):
		variant_info = variant_id.split('_')
		variant_position = int(variant_info[1])
		variant_allele1 = variant_info[3]
		variant_allele2 = variant_info[2]
		fasta_reference_allele = window_sequence[variant_position - window_start]
		if variant_allele1.lower() != fasta_reference_allele.lower() and variant_allele2.lower() != fasta_reference_allele.lower():
			print('allele mismatch eroror')
			pdb.set_trace()
		refs.append(fasta_reference_allele.lower())
		if variant_allele1.lower() == fasta_reference_allele.lower():
			alts.append(variant_allele2.lower())
		else:
			alts.append(variant_allele1.lower())

	return np.asarray(refs), np.asarray(alts)

def extract_snp_sequence_encoding(window_sequence, variant_ids, window_start, flanking_distance=200):
	mapping = {}
	mapping['G'] = 0
	mapping['g'] = 0
	mapping['A'] = 1
	mapping['a'] = 1
	mapping['C'] = 2
	mapping['c'] = 2
	mapping['T'] = 3
	mapping['t'] = 3
	
	encoding = []
	for ii,variant_id in enumerate(variant_ids):
		variant_sequence_encoding = np.zeros((flanking_distance*2 + 1, 4)).astype(int)

		variant_info = variant_id.split('_')
		variant_position = int(variant_info[1])
		variant_allele1 = variant_info[3]
		variant_allele2 = variant_info[2]

		fasta_reference_allele = window_sequence[variant_position - window_start]

		variant_seq = window_sequence[(variant_position-window_start-flanking_distance):(variant_position-window_start+flanking_distance+1)]

		if len(variant_seq) != (flanking_distance*2+1):
			print('assumpition eroror')
			pdb.set_trace()

		# Now convert strings to one hot encoding
		for ii,nucleotide in enumerate(variant_seq):
			variant_sequence_encoding[ii, mapping[nucleotide]] = 1
		
		# Add variant encoding to global vector
		encoding.append(variant_sequence_encoding)

	# Convert to numpy array
	encoding = np.asarray(encoding)

	return encoding


ukbb_preprocessed_for_genome_wide_susie_dir = sys.argv[1]  # Input dir
ldsc_baseline_ld_hg19_annotation_dir = sys.argv[2]  # Input dir (genomic annotations)
output_dir = sys.argv[3]
chrom_num = sys.argv[4]
trait_name = sys.argv[5]
reference_genome_fasta_dir = sys.argv[6]

# First, load in hm3 snps
hm3_snp_file = ldsc_baseline_ld_hg19_annotation_dir + 'baselineLD.' + chrom_num + '.l2.ldscore.gz'
hm3_snps = extract_hm3_snps(hm3_snp_file)

# First, load in genomic annotations of all variants on this chromosome
chrom_annotation_file = ldsc_baseline_ld_hg19_annotation_dir + 'baselineLD.' + chrom_num + '.annot.gz'
annotation_names, variant_to_genomic_annotations = create_variant_to_genomic_annotation_mapping(chrom_annotation_file)


# Input file
input_window_file = ukbb_preprocessed_for_genome_wide_susie_dir + 'genome_wide_susie_windows_and_processed_data_in_sample_ld_chrom_' + chrom_num + '.txt'
# Extract dictionary list of all regression snps. Ie:
### snps that are found in the middle of a window AND are hm3 snps
regression_snps = extract_dictionary_list_of_all_regression_snps(input_window_file, hm3_snps)

# Temporary output file for fasta
temp_fasta_output_file = output_dir + 'temp_fasta_output_' + chrom_num + '.txt'

# Open file handle for output file
output_window_file = output_dir + 'genome_wide_susie_windows_and_non_linear_sldsc_processed_data_in_sample_ld_chrom_' + chrom_num + '.txt'
t = open(output_window_file,'w')
t.write('window_name\tvariant_id\tbeta\tbeta_se\tgenomic_annotation\tmiddle_variant_indices\tregression_variant_indices\tmiddle_regression_variant_indices\tsquared_ld\tregression_snp_squared_ld\tsequence_matrix\treference_alleles\n')

# Open file handle for input file
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
	ref_ld_file = data[9]
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

	# extract trait sample size
	all_sample_sizes = np.loadtxt(sample_size_file)

	# Load in LD matrix
	ref_ld = np.load(ref_ld_file)
	ld_sq = np.square(ref_ld)

	# Load in gwas summary statistics
	# betas
	all_betas = np.loadtxt(beta_file)
	all_betas = all_betas[:, variant_indices]

	# beta std errors
	all_beta_std_err = np.loadtxt(beta_std_err_file)
	all_beta_std_err = all_beta_std_err[:, variant_indices]

	# Ignore windows with nans
	z = all_betas/all_beta_std_err
	if np.sum(np.isnan(z)) > 0:
		continue

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
	# Extract indices of middle variants
	middle_variant_indices, regression_variant_indices = extract_middle_variant_indices(variant_ids, window_start, window_end, regression_snps)
	window_middle_indices = np.intersect1d(regression_variant_indices, middle_variant_indices)

	# Subset LD to just regression snps
	subset_ld_sq = ld_sq[window_middle_indices,:]

	if np.sum(np.isnan(all_betas)) > 0:
		print('skipped')
		continue

	##############################
	# Save data
	##############################
	window_output_root = output_dir + window_name + '_in_sample_ld_'

	# Variant Ids
	variant_id_output_file = window_output_root + 'variant_ids.npy'
	np.save(variant_id_output_file, variant_ids)

	for study_index, study_name in enumerate(all_studies):
		# beta in a specific study
		beta_study_output_file = window_output_root + 'beta_' + study_name + '.npy'
		np.save(beta_study_output_file, (all_betas[study_index,:]).astype('float32'))

		# beta-se in a specific study
		beta_se_study_output_file = window_output_root + 'beta_se_' + study_name + '.npy'
		np.save(beta_se_study_output_file, (all_beta_std_err[study_index,:]).astype('float32'))

	# Use samtooms to get sequence in this window
	buffer_window_start = window_start - 500
	buffer_window_end = window_end + 500
	samtools_fasta_parse_string = 'samtools faidx ' + reference_genome_fasta_dir + 'chr' + str(chrom_num) + '.fa chr' + str(chrom_num) + ':' + str(buffer_window_start) + '-' + str(buffer_window_end) + ' > ' + temp_fasta_output_file
	os.system(samtools_fasta_parse_string)
	window_sequence = get_window_sequence_from_samtools_output(temp_fasta_output_file)

	if 'N' in window_sequence or 'n' in window_sequence:
		print('N in window: skipping for now')
		continue
	if len(window_sequence) != 3001001:
		print('Extract wrong sized window sequence: skipping for now')
		continue

	# Extract reference and alternative alleles
	# QUick error check to make sure variant id reference matches fasta reference
	reference_alleles, alt_alleles = error_check_to_make_sure_variant_id_reference_matches_fasta_reference(window_sequence, buffer_window_start, variant_ids)

	# Extract sequencing encoding for each snp
	sequence_encoding = extract_snp_sequence_encoding(window_sequence, variant_ids, buffer_window_start, flanking_distance=30)


	beta_output_stem = window_output_root + 'beta_'
	beta_se_output_stem = window_output_root + 'beta_se_'

	# genomic annotations
	genomic_anno_output_file = window_output_root + 'genomic_anno.npy'
	np.save(genomic_anno_output_file, genomic_anno_mat)

	# middle variant indices
	middle_window_output_file = window_output_root + 'middle_window_variants.npy'
	np.save(middle_window_output_file, middle_variant_indices)

	# regression variant indices
	regression_indices_window_output_file = window_output_root + 'regression_variants.npy'
	np.save(regression_indices_window_output_file, regression_variant_indices)

	# middle regression variant indices
	middle_regression_indices_window_output_file = window_output_root + 'middle_regression_variants.npy'
	np.save(middle_regression_indices_window_output_file, window_middle_indices)

	# LD matrix
	squared_ld_matrix_output_file = window_output_root + 'squared_ld.npy'
	np.save(squared_ld_matrix_output_file, ld_sq.astype('float32'))

	# regression snp LD matrix
	regression_snp_squared_ld_matrix_output_file = window_output_root + 'regression_snp_squared_ld.npy'
	np.save(regression_snp_squared_ld_matrix_output_file, subset_ld_sq.astype('float32'))

	# Sequence encoding matrix
	sequence_encoding_matrix_output_file = window_output_root + 'sequence_encoding_matrix.npy'
	np.save(sequence_encoding_matrix_output_file, sequence_encoding)

	# Reference alleles
	reference_allele_output_file = window_output_root + 'referene_alleles.npy'
	np.save(reference_allele_output_file, reference_alleles)

	# Print file names to global output file
	t.write(window_name + '\t' + variant_id_output_file + '\t' + beta_output_stem + '\t' + beta_se_output_stem + '\t' + genomic_anno_output_file + '\t' + middle_window_output_file  + '\t' + regression_indices_window_output_file + '\t' + middle_regression_indices_window_output_file + '\t' + squared_ld_matrix_output_file + '\t' + regression_snp_squared_ld_matrix_output_file + '\t' + sequence_encoding_matrix_output_file + '\t' + reference_allele_output_file + '\n')

f.close()
t.close()










