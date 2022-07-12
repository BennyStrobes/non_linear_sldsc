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
	elif training_chromosome_type == 'chrom_6':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 6:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_8':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 8:
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
	elif training_chromosome_type == 'chrom_14':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 14:
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
	middle_hm3_variant_indices = []
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
			middle_hm3_variant_indices.append(data[11])


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
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'srs_inv_file':srs_inv, 's_inv_2_diag_file':s_inv_2_diag, 'D_diag_file': D_diag, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices, 'beta_se_file': beta_se, 'ld_file':ld, 'middle_hm3_variant_indices_file':middle_hm3_variant_indices}
	df = pd.DataFrame(data=dd)

	return df

def marginal_non_linear_sldsc(window_data, samp_size, model_type, temp_output_model_root, max_epochs=5, gradient_steps=1, select_independent_regression_snps=True):
	# Number of annotations
	annotation_data_dimension = get_annotation_data_dimension(window_data)

	# Initialize mapping from annotations to per snp heritability
	genomic_anno_to_gamma_model = initialize_genomic_anno_model(model_type, annotation_data_dimension)
	optimizer = tf.keras.optimizers.Adam()


	# A window is a region of dna space
	# This is number of windows we split dna space
	num_windows = window_data.shape[0]

	# Lopp through windows
	for epoch_iter in range(max_epochs):
		print('epoch iter ' + str(epoch_iter))
		for window_counter, window_iter in enumerate(np.random.permutation(range(num_windows))):
			# Load in data for this window
			window_name = window_data.iloc[window_iter]['window_name']
			print(window_name)
			#window_ld = np.load(window_data.iloc[window_iter]['ld_file'])
			window_beta = np.load(window_data.iloc[window_iter]['beta_file'])
			window_beta_se = np.load(window_data.iloc[window_iter]['beta_se_file'])
			window_chi_sq = np.square(window_beta/window_beta_se)
			window_middle_indices = np.load(window_data.iloc[window_iter]['middle_hm3_variant_indices_file'])
			full_window_middle_indices = np.copy(window_middle_indices)
			#window_middle_indices = np.load(window_data.iloc[window_iter]['middle_variant_indices_file'])
			window_genomic_anno = np.load(window_data.iloc[window_iter]['genomic_annotation_file'])
			squared_ld = np.load(window_data.iloc[window_iter]['squared_ld_file'])
			#middle_indices_weights = np.sum(squared_ld[window_middle_indices,:][:,window_middle_indices],axis=0)
			snp_weights = np.sum(squared_ld,axis=0)
			full_snp_weights = np.copy(snp_weights)

			if select_independent_regression_snps == True:
				snp_weights = snp_weights*0.0 + 1.0

			# Convert to tensors
			window_chi_sq = tf.convert_to_tensor(window_chi_sq.reshape(len(window_chi_sq),1), dtype=tf.float32)
			squared_ld = tf.convert_to_tensor(squared_ld, dtype=tf.float32)
			snp_weights = snp_weights.reshape(len(snp_weights),1)

			if model_type == 'intercept_model':
				window_genomic_anno = np.ones((len(window_beta), 1))

			# If using intercept, alter genomic annotations
			for gradient_iter in range(gradient_steps):
				if select_independent_regression_snps == True:
					window_middle_indices = filter_window_middle_indices_to_independent_snps_at_random(np.asarray(squared_ld)[window_middle_indices,:][:,window_middle_indices], window_middle_indices)
				with tf.GradientTape() as tape:
					window_pred_tau = genomic_anno_to_gamma_model(window_genomic_anno, training=True)
					# Loss value for gradient
					loss_value = ldsc_tf_loss_fxn(window_chi_sq, window_pred_tau, samp_size, squared_ld, window_middle_indices, full_snp_weights)
					if gradient_iter == 0:
						# Loss value for printing
						global_loss_value = ldsc_tf_loss_fxn(window_chi_sq, window_pred_tau, samp_size, squared_ld, full_window_middle_indices, full_snp_weights)
				grads = tape.gradient(loss_value, genomic_anno_to_gamma_model.trainable_weights)
				optimizer.apply_gradients(zip(grads, genomic_anno_to_gamma_model.trainable_weights))


			print('loss')
			print(global_loss_value)
			if np.mod(window_counter,10) == 0:
				print(window_counter)

				print('pred tau')
				print(np.sort(np.asarray(window_pred_tau[:,0])))	

		if np.mod(epoch_iter, 5) == 0:
			genomic_anno_to_gamma_model.save(temp_output_model_root + '_' + str(epoch_iter))


	return genomic_anno_to_gamma_model


def load_in_data(input_dir, trait_name, training_chromosomes):
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
		#middle_variant_indices = np.load(testing_window_data['middle_hm3_variant_indices_file'][window_iter])	
		genomic_anno = np.load(testing_window_data['genomic_annotation_file'][window_iter])	
		marg_beta = np.load(testing_window_data['beta_file'][window_iter])	
		marg_beta_se = np.load(testing_window_data['beta_se_file'][window_iter])
		ld = np.load(testing_window_data['ld_file'][window_iter])
		s_inv_2_diag = np.load(testing_window_data['s_inv_2_diag_file'][window_iter])


		chi_sq = np.square(marg_beta/marg_beta_se)

		if model_type == 'intercept_model':
			genomic_anno_int = np.ones((genomic_anno.shape[0], 1))
			model_pred_tau = genomic_anno_to_gamma_model.predict(genomic_anno_int)[:,0]
		elif model_type == 'sldsc_linear_model' or model_type == 'sldsc_linear_model_non_neg_tau':
			model_pred_tau = np.dot(genomic_anno, genomic_anno_to_gamma_model)
		else:
			model_pred_tau = genomic_anno_to_gamma_model.predict(genomic_anno)[:,0]


		if model_type == 'sldsc_linear_model_non_neg_tau':
			model_pred_tau[model_pred_tau < 0.0] = 0.0
			chi_sq_pred, ld_squared = calculate_chi_sq_pred(model_pred_tau, ld, samp_size, s_inv_2_diag)
		elif model_type == 'sldsc_linear_model':
			chi_sq_pred, ld_squared = calculate_chi_sq_pred(model_pred_tau, ld, samp_size, s_inv_2_diag)
			chi_sq_pred[chi_sq_pred < 0.0] = 1e-6
		else:
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
	print(np.corrcoef(obs_chi_sq, pred_chi_sq))
	return sum_gamma_log_like, se_sum_gamma_log_like

	#print('Average Tau: ' + str(np.mean(pred_tau)))
	#print('Average log like: ' + str(np.mean(gamma_log_like)))
	#print('Corr chi_sq: ' + str(np.corrcoef(obs_chi_sq, pred_chi_sq)))


def ldsc_tf_loss_fxn(chi_sq, pred_tau, samp_size, ld_sq, middle_indices, middle_indices_weights):
	# NEED TO: 
	### 1. GET RID OF NON-MIDDLE INDICES
	### 1. WEIGHT BY LD
	pred_chi_sq = (samp_size*tf.linalg.matmul(ld_sq, pred_tau)) + 1.0

	log_like = (-.5)*tf.math.log(chi_sq) - tf.math.divide(chi_sq, 2.0*pred_chi_sq) - (.5*tf.math.log(2.0*pred_chi_sq))

	middle_indices_log_like = tf.gather(log_like, middle_indices, axis=0)

	weighted_middle_indices_log_like = tf.math.divide(middle_indices_log_like, middle_indices_weights)

	return -tf.math.reduce_sum(weighted_middle_indices_log_like)/np.sum(1.0/middle_indices_weights)

def ldsc_mse_tf_loss_fxn(chi_sq, pred_tau, samp_size, ld_sq, middle_indices, middle_indices_weights):
	# NEED TO: 
	### 1. GET RID OF NON-MIDDLE INDICES
	### 1. WEIGHT BY LD
	pred_chi_sq = (samp_size*tf.linalg.matmul(ld_sq, pred_tau)) + 1.0


	squared_error = tf.pow(pred_chi_sq - chi_sq,2)
	middle_indices_squared_error = tf.gather(squared_error, middle_indices, axis=0)
	weighted_middle_indices_squared_error = tf.math.divide(middle_indices_squared_error, middle_indices_weights)

	return tf.math.reduce_sum(weighted_middle_indices_squared_error)/np.sum(1.0/middle_indices_weights)
def ldsc_mse_tf_loss_fxn_unweighted(chi_sq, pred_tau, samp_size, ld_sq, middle_indices, middle_indices_weights):
	# NEED TO: 
	### 1. GET RID OF NON-MIDDLE INDICES
	### 1. WEIGHT BY LD
	pred_chi_sq = (samp_size*tf.linalg.matmul(ld_sq, pred_tau)) + 1.0


	squared_error = tf.pow(pred_chi_sq - chi_sq,2)
	middle_indices_squared_error = tf.gather(squared_error, middle_indices, axis=0)

	return tf.math.reduce_mean(middle_indices_squared_error)

def get_non_negative_indices_from_sldsc_run(file_name):
	dicti = {}
	f = open(file_name)
	head_count = 0
	counter = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		if float(data[3]) != 1e-6:
			dicti[counter] = 1
		counter = counter + 1
	f.close()
	return dicti

def get_non_neg_indices_in_standard_output_file(input_file, output_file, non_neg_indices_dicti):
	f = open(input_file)
	t = open(output_file,'w')
	head_count = 0
	counter = 0
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
			t.write(line + '\n')
			continue

		if counter in non_neg_indices_dicti:
			t.write(line + '\n')
		counter = counter + 1
	f.close()
	t.close()


trait_name = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
samp_size = float(sys.argv[4])
testing_chromosome_type = sys.argv[5]




testing_chromosomes = get_training_chromosomes(testing_chromosome_type)
testing_window_data = load_in_testing_data(input_dir, trait_name, testing_chromosomes)

# Model training
training_chromosome_type = 'even'




model_vectors = ['sldsc_linear_model_non_neg_tau', 'sldsc_linear_model', 'neural_network_no_drops', 'linear_model', 'intercept_model']


summary_output_file = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_results_training_data_1_grad_steps_False_independent_reg_snps_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval.txt'

t = open(summary_output_file,'w')
t.write('model\tsum_gamma_log_like\tse_sum_gamma_log_like\n')

for model_type in model_vectors:
	for itera in [120]:

		print('\n')	
		print('#######################')
		print('#######################')
		print(model_type)
		print(itera)


		if model_type == 'sldsc_linear_model' or model_type == 'sldsc_linear_model_non_neg_tau':
			model_root = output_dir + trait_name + '_sldsc_source_code_even_chrom_no_intercept.results'
			data = np.loadtxt(model_root, dtype=str, delimiter='\t')
			genomic_anno_to_gamma_model = data[1:,7].astype(float)
		else:

			#itera = '70'
			if model_type == 'neural_network_no_drops':
				itera = 100
			model_root = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates4_results_training_data_' + model_type + '_even_annotations_to_gamma_model_temp_' + str(itera)
			#model_root = output_dir + trait_name + '_nonlinear_sldsc_matrix_inversion_mse_updates_results_training_data_' + model_type + '_' + '1e-3' + '_diagonal_padding_' + training_chromosome_type + '_annotations_to_gamma_model'
			#model_root = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_results_training_data_' + model_type + '_1_grad_steps_False_independent_reg_snps_' + training_chromosome_type + '_annotations_to_gamma_model_temp_' + str(itera)
			# Save training data results
			# Save TensorFlow model
			if os.path.isdir(model_root) == False:
				continue
			genomic_anno_to_gamma_model = tf.keras.models.load_model(model_root,custom_objects={'ldsc_tf_loss_fxn':ldsc_tf_loss_fxn})


		testing_ldsc_likelihood_output_file = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_results_training_data_' + model_type + '_1_grad_steps_False_independent_reg_snps_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval_temper.txt'
		run_ld_score_regression_likelihood_evaluation(testing_window_data, genomic_anno_to_gamma_model, samp_size, testing_ldsc_likelihood_output_file, model_type)
		sum_gamma_log_like, se_sum_gamma_log_like = likelihood_summary(testing_ldsc_likelihood_output_file)
		t.write(model_type + '\t' + str(sum_gamma_log_like) + '\t' + str(se_sum_gamma_log_like) + '\n')
t.close()




# GET NON-negative indices
summary_non_neg_output_file = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_results_training_data_1_grad_steps_False_independent_reg_snps_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval_non_negative.txt'

t2 = open(summary_non_neg_output_file,'w')
t2.write('model\tsum_gamma_log_like\tse_sum_gamma_log_like\n')



model_type = 'sldsc_linear_model'
testing_ldsc_likelihood_output_file = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_results_training_data_' + model_type + '_1_grad_steps_False_independent_reg_snps_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval_temper.txt'
non_neg_indices_dicti = get_non_negative_indices_from_sldsc_run(testing_ldsc_likelihood_output_file)

for model_type in model_vectors:
	for itera in [120]:
		testing_ldsc_likelihood_output_file = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_results_training_data_' + model_type + '_1_grad_steps_False_independent_reg_snps_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval_temper.txt'
		testing_ldsc_likelihood_non_neg_output_file = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_results_training_data_' + model_type + '_1_grad_steps_False_independent_reg_snps_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval_non_neg_temper.txt'
		get_non_neg_indices_in_standard_output_file(testing_ldsc_likelihood_output_file, testing_ldsc_likelihood_non_neg_output_file, non_neg_indices_dicti)
		sum_gamma_log_like, se_sum_gamma_log_like = likelihood_summary(testing_ldsc_likelihood_non_neg_output_file)
		t2.write(model_type + '\t' + str(sum_gamma_log_like) + '\t' + str(se_sum_gamma_log_like) + '\n')

t2.close()