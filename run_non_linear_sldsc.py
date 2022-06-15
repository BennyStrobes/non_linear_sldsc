import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas as pd
import os
import pdb
import tensorflow as tf
import gzip
import time




def get_training_chromosomes(training_chromosome_type):
	if training_chromosome_type == 'even':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if np.mod(chrom_num,2) == 0:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_5':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 5:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosomes == 'odd':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if np.mod(chrom_num,2) != 0:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes		


def load_in_data(input_dir, trait_name, training_chromosomes):
	window_names = []
	variant_id = []
	srs_inv = []
	s_inv_2_diag = []
	D_diag = []
	beta = []
	genomic_annotation = []
	middle_variant_indices = []
	# Initialize global dictionary to keep track of current estimates of beta_mu and beta_var
	window_to_beta_mu = {}
	window_to_beta_var = {}
	window_to_gamma = {}
	for chrom_num in range(1,23):
		if chrom_num not in training_chromosomes:
			continue
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
			window_names.append(data[0])
			variant_id.append(data[1])
			srs_inv.append(data[2])
			s_inv_2_diag.append(data[3])
			D_diag.append(data[4])
			beta.append(data[5])
			genomic_annotation.append(data[6])
			middle_variant_indices.append(data[7])

			beta_data = np.load(data[5])
			num_snps = len(beta_data)
			window_to_beta_mu[window_name] = np.zeros(num_snps)
			window_to_beta_var[window_name] = np.ones(num_snps)
			window_to_gamma[window_name] = np.ones(num_snps)
		f.close()
	# Quick error checking
	if len(np.unique(window_names)) != len(window_names):
		print('assumption error')
		pdb.set_trace()
	# Put data in pandas df
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'srs_inv_file':srs_inv, 's_inv_2_diag_file':s_inv_2_diag, 'D_diag_file': D_diag, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices}
	df = pd.DataFrame(data=dd)

	return df, window_to_beta_mu, window_to_beta_var, window_to_gamma

def update_beta_distributions(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma):
	# Get number of windows
	num_windows = window_data.shape[0]
	print(num_windows)

	# Now loop through windows
	for window_iter in range(num_windows):
		print(window_iter)
		# Extract relevent info for this window
		window_name = window_data['window_name'][window_iter]
		window_srs_inv = np.load(window_data['srs_inv_file'][window_iter])
		window_s_inv_2_diag = np.load(window_data['s_inv_2_diag_file'][window_iter])
		window_D_diag = np.load(window_data['D_diag_file'][window_iter])
		window_marginal_betas = np.load(window_data['beta_file'][window_iter])

		beta_mu = window_to_beta_mu[window_name]
		beta_var = window_to_beta_var[window_name]
		gamma = window_to_gamma[window_name]

		# UPDATES for this window
		num_snps = len(beta_mu)
		# Marginal betas will all effects removed
		residual = window_marginal_betas - np.dot(window_srs_inv, beta_mu)

		# Loop through snps
		for k_index in range(num_snps):

			# get marginal betas with all effects removed other than the snp of interest
			residual = residual + window_srs_inv[:, k_index]*beta_mu[k_index]

			# Calculate terms involved in the update
			b_term = residual[k_index]*window_s_inv_2_diag[k_index]
			a_term = (-.5*window_D_diag[k_index]) - (.5*gamma[k_index])

			# VI Updates
			beta_var[k_index] = -1.0/(2.0*a_term)
			beta_mu[k_index] = b_term*beta_var[k_index]

			# Update resid for next round (after this resid includes effects of all genes)
			residual = residual - window_srs_inv[:,k_index]*beta_mu[k_index]
		window_to_beta_mu[window_name] = beta_mu
		window_to_beta_var[window_name] = beta_var
	return window_to_beta_mu, window_to_beta_var

def extract_non_linear_function_training_data(window_data, window_to_beta_mu, window_to_beta_var):
	# Initialize output data
	beta_mu_train = []
	beta_var_train = []
	genomic_anno_train = []

	# Get number of windows
	num_windows = window_data.shape[0]
	print(num_windows)

	# Now loop through windows
	for window_iter in range(num_windows):
		# Extract relevent info for this window
		window_name = window_data['window_name'][window_iter]
		genomic_anno = np.load(window_data['genomic_annotation_file'][window_iter])
		middle_indices = np.load(window_data['middle_variant_indices_file'][window_iter])

		# Add to training data objects
		beta_mu_train.append(window_to_beta_mu[window_name][middle_indices])
		beta_var_train.append(window_to_beta_var[window_name][middle_indices])
		genomic_anno_train.append(genomic_anno[middle_indices,:])
	return np.hstack(beta_mu_train), np.hstack(beta_var_train), np.vstack(genomic_anno_train)

def non_linear_sldsc(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma, output_root, max_iterations=200):
	# VI Iterations
	for vi_iter in range(max_iterations):
		# Part 1 Update window_to_beta_mu and window_to_beta_mu given current values of window_to_gamma
		window_to_beta_mu, window_to_beta_var = update_beta_distributions(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma)

		# Part 2: Extract beta_mu_training, beta_var_training, and genomic_anno_training for all snps in middle of a window
		beta_mu_train, beta_var_train, genome_anno_train = extract_non_linear_function_training_data(window_data, window_to_beta_mu, window_to_beta_var)

		# TEMP FOR CODE DEV
		beta_mu_output_file = output_root + 'beta_mu_temp.npy'
		np.save(beta_mu_output_file, beta_mu_train)
		beta_var_output_file = output_root + 'beta_var_temp.npy'
		np.save(beta_var_output_file, beta_var_train)
		genome_anno_output_file = output_root + 'genome_anno_temp.npy'
		np.save(genome_anno_output_file, genome_anno_train)
		pdb.set_trace()


		# Part 3: Update non-linear mapping from genomic annotations to gamma

		# Part 4: Update window_to_gamma given updated non-linear mapping 




trait_name = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]



training_chromosome_type = 'chrom_5'
training_chromosomes = get_training_chromosomes(training_chromosome_type)




# load in data
window_data, window_to_beta_mu, window_to_gamma, window_to_beta_var = load_in_data(input_dir, trait_name, training_chromosomes)


output_root = output_dir + trait_name + '_nonlinear_sldsc_results_training_data_' + training_chromosome_type
non_linear_sldsc(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma, output_root)



pdb.set_trace()
