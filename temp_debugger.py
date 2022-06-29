import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas as pd
import os
import pdb
import tensorflow as tf
import gzip
import time
from numba import njit, prange
from joblib import Parallel, delayed
from scipy.stats import gamma




def get_training_chromosomes(training_chromosome_type):
	if training_chromosome_type == 'even':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if np.mod(chrom_num,2) == 0:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_2_3':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 2 or chrom_num == 3:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_2':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 2:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_1':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 1:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_3':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 3:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_5':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 5:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_7':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 7:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_9':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 9:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_15':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 15:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_1_to_7':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num < 8:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_6_8_10':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 6 or chrom_num == 8 or chrom_num == 10:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_8_10_12_14':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 12 or chrom_num == 8 or chrom_num == 10 or chrom_num == 14:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'odd':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if np.mod(chrom_num,2) != 0:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes		

def load_in_testing_data(input_dir, trait_name, training_chromosomes):
	window_names = []
	variant_id = []
	srs_inv = []
	s_inv_2_diag = []
	D_diag = []
	beta = []
	beta_se = []
	ld = []
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
			beta_se.append(data[8])
			ld.append(data[9])

			beta_data = np.load(data[5])
			num_snps = len(beta_data)
			#window_to_beta_mu[window_name] = np.zeros(num_snps)
			#window_to_beta_var[window_name] = np.ones(num_snps)
			#window_to_gamma[window_name] = np.ones(num_snps)*1e-5
		f.close()
	# Quick error checking
	if len(np.unique(window_names)) != len(window_names):
		print('assumption error')
		pdb.set_trace()
	# Put data in pandas df
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'srs_inv_file':srs_inv, 's_inv_2_diag_file':s_inv_2_diag, 'D_diag_file': D_diag, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices, 'beta_se_file': beta_se, 'ld_file':ld}
	df = pd.DataFrame(data=dd)

	return df


def load_in_data(input_dir, trait_name, training_chromosomes):
	window_names = []
	variant_id = []
	srs_inv = []
	s_inv_2_diag = []
	D_diag = []
	beta = []
	beta_se = []
	ld = []
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
			beta_se.append(data[8])
			ld.append(data[9])

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
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'srs_inv_file':srs_inv, 's_inv_2_diag_file':s_inv_2_diag, 'D_diag_file': D_diag, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices, 'beta_se_file': beta_se, 'ld_file':ld}
	df = pd.DataFrame(data=dd)

	return df, window_to_beta_mu, window_to_beta_var, window_to_gamma

def gaussian_neg_log_likelihood_tf_padded_loss(beta_squared_true, gamma_pred):
	epsilon = 1e-30
	nll = tf.math.log(gamma_pred + epsilon) + tf.math.divide(beta_squared_true, (gamma_pred+epsilon))
	return nll



def calculate_chi_sq_pred(model_pred_tau, ld, samp_size, s_inv_2_diag):
	ld_sq = np.square(ld)

	chi_sq_pred = []
	# loop through variants
	for ii in range(len(model_pred_tau)):
		pred = 1.0 + (samp_size*np.sum(ld_sq[ii,:]*model_pred_tau))
		chi_sq_pred.append(pred)
	return np.asarray(chi_sq_pred), ld_sq

def calculate_ld_score_log_likelihood(chi_sq, chi_sq_pred, ld_scores):
	log_likes = gamma.logpdf(chi_sq, .5, scale=2*chi_sq_pred)/ld_scores
	return np.asarray(log_likes)


def run_ld_score_regression_likelihood_evaluation(testing_window_data, genomic_anno_to_gamma_model, samp_size, output_file, model_type):
	# Get number of windows
	num_windows = testing_window_data.shape[0]

	t = open(output_file,'w')
	t.write('snp\twindow\tobserved_chi_sq\tpred_chi_sq\tldsc_gamma_log_like\tpred_tau\tld_score\n')
	# Now loop through windows
	for window_iter in range(num_windows):
		# Extract relevent info for this window
		window_name = testing_window_data['window_name'][window_iter]
		variant_ids = np.load(testing_window_data['variant_id_file'][window_iter])	

		middle_variant_indices = np.load(testing_window_data['middle_variant_indices_file'][window_iter])	
		genomic_anno = np.load(testing_window_data['genomic_annotation_file'][window_iter])	
		marg_beta = np.load(testing_window_data['beta_file'][window_iter])	
		marg_beta_se = np.load(testing_window_data['beta_se_file'][window_iter])
		ld = np.load(testing_window_data['ld_file'][window_iter])
		s_inv_2_diag = np.load(testing_window_data['s_inv_2_diag_file'][window_iter])


		chi_sq = np.square(marg_beta/marg_beta_se)

		if model_type == 'intercept_model':
			genomic_anno_int = np.ones((genomic_anno.shape[0], 1))
			model_pred_tau = genomic_anno_to_gamma_model.predict(genomic_anno_int)[:,0]
		else:
			model_pred_tau = genomic_anno_to_gamma_model.predict(genomic_anno)[:,0]
		
		chi_sq_pred, ld_squared = calculate_chi_sq_pred(model_pred_tau, ld, samp_size, s_inv_2_diag)

		ld_scores = np.sum(ld_squared,axis=0)

		# Filter data to middle variants
		#ld = ld[middle_variant_indices,:][:, middle_variant_indices]
		#genomic_anno = genomic_anno[middle_variant_indices,:]
		chi_sq = chi_sq[middle_variant_indices]
		chi_sq_pred = chi_sq_pred[middle_variant_indices]
		ld_scores = ld_scores[middle_variant_indices]
		variant_ids = variant_ids[middle_variant_indices]
		model_pred_tau = model_pred_tau[middle_variant_indices]

		log_likelihoods = calculate_ld_score_log_likelihood(chi_sq, chi_sq_pred, ld_scores)


		for ii in range(len(log_likelihoods)):
			t.write(variant_ids[ii] + '\t' + window_name + '\t' + str(chi_sq[ii]) + '\t' + str(chi_sq_pred[ii]) + '\t' + str(log_likelihoods[ii]) + '\t' + str(model_pred_tau[ii]) + '\t' + str(ld_scores[ii]) + '\n')
	t.close()

def bootstrap_gamma_log_like(gamma_log_like, num_bootstrap_samples):
	sum_gamma_log_like = np.sum(gamma_log_like)
	num_samples = len(gamma_log_like)
	bootstrap_arr = []
	for bootstrap_sample in np.arange(num_bootstrap_samples):
		bootstrap_sample_indices = np.random.choice(np.arange(num_samples), size=num_samples)
		bootstrap_sum_gamma_log_like = np.sum(gamma_log_like[bootstrap_sample_indices])
		bootstrap_arr.append(bootstrap_sum_gamma_log_like)
	bootstrap_arr=np.asarray(bootstrap_arr)
	sum_gamma_log_like_bootstrap_se = np.sqrt(np.sum(np.square(bootstrap_arr-np.mean(bootstrap_arr)))/(num_bootstrap_samples-1))
	return sum_gamma_log_like, sum_gamma_log_like_bootstrap_se

def likelihood_summary(testing_ldsc_likelihood_output_file):
	aa = np.loadtxt(testing_ldsc_likelihood_output_file,dtype=str,delimiter='\t')
	pred_tau = aa[1:,5].astype(float)
	obs_chi_sq = aa[1:,2].astype(float)
	pred_chi_sq = aa[1:,3].astype(float)
	gamma_log_like = aa[1:,4].astype(float)

	num_bootstrap_samples = 1000
	sum_gamma_log_like, se_sum_gamma_log_like = bootstrap_gamma_log_like(gamma_log_like, num_bootstrap_samples)
	print(sum_gamma_log_like)
	print(se_sum_gamma_log_like)
	return sum_gamma_log_like, se_sum_gamma_log_like

	#print('Average Tau: ' + str(np.mean(pred_tau)))
	#print('Average log like: ' + str(np.mean(gamma_log_like)))
	#print('Corr chi_sq: ' + str(np.corrcoef(obs_chi_sq, pred_chi_sq)))


trait_name = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
samp_size = float(sys.argv[4])

print(trait_name)






testing_chromosome_type = 'chrom_2_3'
testing_chromosome_type = 'chrom_5'
testing_chromosome_type = 'even'

testing_chromosomes = get_training_chromosomes(testing_chromosome_type)
testing_window_data = load_in_testing_data(input_dir, trait_name, testing_chromosomes)

# Model training
training_chromosome_type = 'even'




model_vectors = [('neural_network_no_drops', '10', 'v3'), ('neural_network', '10', 'v4'), ('linear_model', '10', 'v2'), ('intercept_model', '15', 'v2')]
model_vectors = [('neural_network_no_drops', '10', 'v3'), ('linear_model', '10', 'v2'), ('intercept_model', '15', 'v2')]


summary_output_file = output_dir + trait_name + '_nonlinear_sldsc_multivariate_results_training_data_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval.txt'
t = open(summary_output_file,'w')
t.write('model\tsum_gamma_log_like\tse_sum_gamma_log_like\n')


for model_tuple in model_vectors:

	model_type = model_tuple[0]
	model_iter = model_tuple[1]
	model_version = model_tuple[2]

	print('\n')	
	print('#######################')
	print('#######################')
	print(model_type + '    /   ' + str(model_iter))



	temp_output_model_root = output_dir + trait_name + '_nonlinear_sldsc_univariate_' + model_version + '_updates_results_training_data_' + model_type + '_' + training_chromosome_type + '_annotations_to_gamma_model_temp_' + model_iter

	# Save training data results
	# Save TensorFlow model
	#genomic_anno_to_gamma_model = tf.keras.models.load_model(temp_output_model_root,custom_objects={'gaussian_neg_log_likelihood_tf_padded_loss':gaussian_neg_log_likelihood_tf_padded_loss})


	testing_ldsc_likelihood_output_file = output_dir + trait_name + '_nonlinear_sldsc_multivariate_results_training_data_' + model_type + '_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval_temper_' + model_iter + '.txt'
	#run_ld_score_regression_likelihood_evaluation(testing_window_data, genomic_anno_to_gamma_model, samp_size, testing_ldsc_likelihood_output_file, model_type)
	sum_gamma_log_like, se_sum_gamma_log_like = likelihood_summary(testing_ldsc_likelihood_output_file)

	t.write(model_type + '\t' + str(sum_gamma_log_like) + '\t' + str(se_sum_gamma_log_like) + '\n')

t.close()


