import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas as pd
import os
import pdb
import time


def window_includes_mhc_region(window_chrom_num, window_start, window_end, mhc_region):
	if window_chrom_num != '6':
		return False
	else:
		in_mhc = False
		for pos in range(window_start,window_end):
			if pos in mhc_region:
				in_mhc = True
		return in_mhc


def load_in_all_data(input_dir, trait_name):
	window_names = []
	variant_id = []
	srs_inv = []
	s_inv_2_diag = []
	D_mat = []
	D_diag = []

	beta = []
	beta_se = []
	ld = []
	squared_ld = []
	genomic_annotation = []
	middle_variant_indices = []
	middle_hm3_variant_indices = []
	# Initialize global dictionary to keep track of current estimates of beta_mu and beta_var
	window_to_beta_mu = {}
	window_to_beta_var = {}
	window_to_gamma = {}

	mhc_region = {}
	for pos in range(25726063,33400644):
		mhc_region[pos] = 1


	for chrom_num in range(1,23):
		input_file = input_dir + trait_name + '_genome_wide_susie_windows_and_non_linear_sldsc_processed_data_chrom_' + str(chrom_num) + '.txt'
		head_count = 0
		f = open(input_file)
		for line in f:
			line = line.rstrip()
			data = line.split('\t')
			if head_count == 0:
				head_count = head_count + 1
				continue
			window_name = data[0]

			# Ignore regions in MHC
			window_chrom_num = window_name.split(':')[0]
			window_start = int(window_name.split(':')[1])
			window_end = int(window_name.split(':')[2])
			if window_includes_mhc_region(window_chrom_num, window_start, window_end, mhc_region):
				continue

			window_names.append(data[0])
			variant_id.append(data[1])
			srs_inv.append(data[2])
			s_inv_2_diag.append(data[3])
			D_mat.append(data[10])
			D_diag.append(data[4])
			beta.append(data[5])
			genomic_annotation.append(data[6])
			middle_variant_indices.append(data[7])
			beta_se.append(data[8])
			ld.append(data[9])
			middle_hm3_variant_indices.append(data[11])
			squared_ld.append(data[12])

			beta_data = np.load(data[5])
			num_snps = len(beta_data)
			window_to_beta_mu[window_name] = np.zeros(num_snps)
			window_to_beta_var[window_name] = np.ones(num_snps)
			window_to_gamma[window_name] = np.ones(num_snps)*1e-7
		f.close()
	# Quick error checking
	if len(np.unique(window_names)) != len(window_names):
		print('assumption error')
		pdb.set_trace()
	# Put data in pandas df
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'srs_inv_file':srs_inv, 's_inv_2_diag_file':s_inv_2_diag, 'D_mat_file': D_mat, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices, 'middle_hm3_variant_indices_file':middle_hm3_variant_indices, 'beta_se_file': beta_se, 'ld_file':ld, 'D_diag_file': D_diag, 'squared_ld_file': squared_ld}
	df = pd.DataFrame(data=dd)

	return df




########################
# Command line args
########################
trait_name = sys.argv[1]
preprocessed_data_for_non_linear_sldsc_dir = sys.argv[2]
samp_size = int(sys.argv[3])
diagonal_padding_str = (sys.argv[4])
job_number = int(sys.argv[5])
num_jobs = int(sys.argv[6])

diagonal_padding = float(diagonal_padding_str)

# Extract window data
window_data = load_in_all_data(preprocessed_data_for_non_linear_sldsc_dir, trait_name)

# Total number of windows
num_windows = window_data.shape[0]

# Parallelization stuff
windows_per_job = np.int(np.ceil(num_windows/num_jobs))
job_start = windows_per_job*job_number
job_end = windows_per_job*(job_number+1)
if job_end > num_windows:
	job_end = num_windows




# Loop through windows  (for this job)

for window_iter in np.arange(job_start, job_end):
	start_time = time.time()

	window_name = window_data.iloc[window_iter]['window_name']
	print(window_name)
	window_beta = np.load(window_data.iloc[window_iter]['beta_file'])
	window_beta_se = np.load(window_data.iloc[window_iter]['beta_se_file'])
	window_chi_sq = np.square(window_beta/window_beta_se)
	squared_ld = np.load(window_data.iloc[window_iter]['squared_ld_file'])
	window_middle_indices = np.load(window_data.iloc[window_iter]['middle_variant_indices_file'])


	# Invert squared ld
	squared_ld_invert = np.linalg.inv(squared_ld + (diagonal_padding*np.eye(squared_ld.shape[0])))

	# Decouple chi squared stats
	temp = (window_chi_sq - 1.0)/samp_size
	final = np.dot(squared_ld_invert, temp)

	print(np.mean(final[window_middle_indices]))


	output_file = preprocessed_data_for_non_linear_sldsc_dir + trait_name + '_' + window_name + '_' + diagonal_padding_str + '_decorrelated_chi_squared.npy'
	np.save(output_file, final)
	end_time = time.time()
	print(end_time-start_time)

