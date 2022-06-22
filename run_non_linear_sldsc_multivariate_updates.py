import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas as pd
import os
import pdb
import tensorflow as tf
import gzip
import time
from numba import njit, prange, jit
from joblib import Parallel, delayed
from scipy.stats import gamma
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from joblib.externals.loky import set_loky_pickler

from joblib import wrap_non_picklable_objects


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
	elif training_chromosome_type == 'chrom_5':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 5:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_1_18':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num <= 18:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_19_22':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num > 19:
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

def window_includes_mhc_region(window_chrom_num, window_start, window_end, mhc_region):
	if window_chrom_num != '6':
		return False
	else:
		in_mhc = False
		for pos in range(window_start,window_end):
			if pos in mhc_region:
				in_mhc = True
		return in_mhc


def load_in_data(input_dir, trait_name, training_chromosomes):
	window_names = []
	variant_id = []
	srs_inv = []
	s_inv_2_diag = []
	D_mat = []
	beta = []
	beta_se = []
	ld = []
	genomic_annotation = []
	middle_variant_indices = []
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
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'srs_inv_file':srs_inv, 's_inv_2_diag_file':s_inv_2_diag, 'D_mat_file': D_mat, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices, 'beta_se_file': beta_se, 'ld_file':ld}
	df = pd.DataFrame(data=dd)

	return df, window_to_beta_mu, window_to_beta_var, window_to_gamma

def update_gamma_distributions(window_data, window_to_gamma, genomic_anno_to_gamma_model):
	# Get number of windows
	num_windows = window_data.shape[0]
	#print(num_windows)

	# Now loop through windows
	for window_iter in range(num_windows):
		#print(window_iter)
		window_name = window_data['window_name'][window_iter]
		genomic_anno = np.load(window_data['genomic_annotation_file'][window_iter])
		window_gamma = genomic_anno_to_gamma_model.predict(genomic_anno)
		window_to_gamma[window_name] = window_gamma[:,0]

	return window_to_gamma

@jit
def multivariate_updates(gamma, window_D_mat, window_s_inv_2_diag, window_marginal_betas):
	covariance = np.linalg.inv(np.diag(1.0/gamma) + window_D_mat)
	mu = np.dot(covariance, window_s_inv_2_diag*window_marginal_betas)
	return mu, covariance

def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def relu_np(x): return np.maximum(x, 0)

@wrap_non_picklable_objects
def update_beta_distribution_in_single_window(window_name, window_s_inv_2_diag_file, window_D_mat_file, window_marginal_betas_file, window_genomic_annotation_file, window_middle_variant_indices_file, vi_iter, weights, model_type):
	#window_srs_inv = np.load(window_srs_inv_file)
	window_s_inv_2_diag = np.load(window_s_inv_2_diag_file)
	window_D_mat = np.load(window_D_mat_file)
	window_marginal_betas = np.load(window_marginal_betas_file)
	genomic_anno = np.load(window_genomic_annotation_file)


	#diff=genomic_anno_to_gamma_model.predict(genomic_anno) - tf.math.softplus(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(genomic_anno, weights[0]) + weights[1]), weights[2]) + weights[3]), weights[4]) + weights[5])
	if vi_iter < 2:
		gamma = np.ones(len(window_marginal_betas))*1e-7
	else:
		#gamma = weights.predict(genomic_anno)
		if model_type == 'neural_network':
			gamma = softplus_np(np.dot(relu_np(np.dot(relu_np(np.dot(genomic_anno, weights[0]) + weights[1]), weights[2]) + weights[3]), weights[4]) + weights[5])
			gamma = gamma[:,0]
		elif model_type == 'linear_model':
			gamma = softplus_np(np.dot(genomic_anno, weights[0]) + weights[1])
			gamma = gamma[:,0]
		elif model_type == 'intercept_model':
			genomic_anno = np.ones((genomic_anno.shape[0],1))
			gamma = softplus_np(np.dot(genomic_anno, weights[0]) + weights[1])
			gamma = gamma[:,0]

	beta_mu, beta_covariance = multivariate_updates(gamma, window_D_mat, window_s_inv_2_diag, window_marginal_betas)
	beta_var = np.diag(beta_covariance)

	beta_squared = np.square(beta_mu) + beta_var

	
	middle_indices = np.load(window_middle_variant_indices_file)

	# Add to training data objects
	beta_squared_train = beta_squared[middle_indices]
	genomic_anno_train = genomic_anno[middle_indices,:]


	return (beta_squared_train, genomic_anno_train)


def fast_update_beta_distributions(window_data, genomic_anno_to_gamma_model, vi_iter, subset_iter, model_type, parallel_bool=True):
	# Get number of windows
	num_windows = window_data.shape[0]

	weights = genomic_anno_to_gamma_model.get_weights()

	if parallel_bool == False:
		beta_data = []
		# Now loop through windows
		for window_iter in range(num_windows):
			beta_data.append(update_beta_distribution_in_single_window(window_data.iloc[window_iter]['window_name'], window_data.iloc[window_iter]['s_inv_2_diag_file'], window_data.iloc[window_iter]['D_mat_file'], window_data.iloc[window_iter]['beta_file'], window_data.iloc[window_iter]['genomic_annotation_file'], window_data.iloc[window_iter]['middle_variant_indices_file'], vi_iter, weights, model_type))
	elif parallel_bool == True:
		beta_data = Parallel(n_jobs=20)(delayed(update_beta_distribution_in_single_window)(window_data.iloc[window_iter]['window_name'], window_data.iloc[window_iter]['s_inv_2_diag_file'], window_data.iloc[window_iter]['D_mat_file'], window_data.iloc[window_iter]['beta_file'], window_data.iloc[window_iter]['genomic_annotation_file'], window_data.iloc[window_iter]['middle_variant_indices_file'], vi_iter, weights, model_type) for window_iter in range(num_windows))
		#beta_data = Parallel(n_jobs=20)(delayed(update_beta_distribution_in_single_window)(window_data['window_name'][window_iter], window_data['srs_inv_file'][window_iter], window_data['s_inv_2_diag_file'][window_iter], window_data['D_mat_file'][window_iter], window_data['beta_file'][window_iter], window_data['genomic_annotation_file'][window_iter], window_data['middle_variant_indices_file'][window_iter], genomic_anno_to_gamma_model, vi_iter) for window_iter in range(num_windows))


	# GET ORGANIZED NEURAL NET TRAINING DATA
	beta_squared_train = []
	genomic_anno_train = []
	for window_iter in range(num_windows):
		# Extract relevent info for this window
		window_name = window_data.iloc[window_iter]['window_name']
		beta_squared_train.append(beta_data[window_iter][0])
		genomic_anno_train.append(beta_data[window_iter][1])

	return np.hstack(beta_squared_train), np.vstack(genomic_anno_train) 



def extract_non_linear_function_training_data(window_data, window_to_beta_mu, window_to_beta_var):
	# Initialize output data
	beta_mu_train = []
	beta_var_train = []
	genomic_anno_train = []

	# Get number of windows
	num_windows = window_data.shape[0]
	#print(num_windows)

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


def save_snp_effects_on_training_data_to_output_file(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma, output_file):
	# Open output file handle 
	t = open(output_file,'w')
	t.write('snp_id\twindow\tmiddle_snp_boolean\tbeta_mu\tbeta_var\tgamma\n')

	# Get number of windows
	num_windows = window_data.shape[0]


	# Now loop through windows
	for window_iter in range(num_windows):
		# Extract relevent info for this window
		window_name = window_data['window_name'][window_iter]
		ordered_variants = np.load(window_data['variant_id_file'][window_iter])
		middle_variant_indices = np.load(window_data['middle_variant_indices_file'][window_iter])

		middle_variants = {}
		for variant_id in ordered_variants[middle_variant_indices]:
			middle_variants[variant_id] = 1

		window_beta_mu = window_to_beta_mu[window_name]
		window_beta_var = window_to_beta_var[window_name]
		window_gamma = window_to_gamma[window_name]

		for ii, variant_id in enumerate(ordered_variants):
			middle_variant_bool = 'False'
			if variant_id in middle_variants:
				middle_variant_bool = 'True'
			t.write(variant_id + '\t' + window_name + '\t' + middle_variant_bool + '\t' + str(window_beta_mu[ii]) + '\t' + str(window_beta_var[ii]) + '\t' + str(window_gamma[ii]) + '\n')
	t.close()



def gaussian_neg_log_likelihood_tf_loss(beta_squared_true, gamma_pred):
	nll = tf.math.log(gamma_pred) + tf.math.divide(beta_squared_true, gamma_pred)
	return nll

def gaussian_neg_log_likelihood_tf_padded_loss(beta_squared_true, gamma_pred):
	epsilon = 1e-30
	nll = tf.math.log(gamma_pred + epsilon) + tf.math.divide(beta_squared_true, (gamma_pred+epsilon))
	return nll


def init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=64, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	model.compile(loss=gaussian_neg_log_likelihood_tf_padded_loss, optimizer='adam')

	return model

def init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=1, activation='softplus', input_dim=annotation_data_dimension))

	model.compile(loss=gaussian_neg_log_likelihood_tf_padded_loss, optimizer='adam')

	return model

def get_annotation_data_dimension(window_data):
	genomic_anno_dim = np.load(window_data['genomic_annotation_file'][0]).shape[1]
	return genomic_anno_dim


def non_linear_sldsc(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma, model_type, temp_output_model_root, max_iterations=200):
	
	annotation_data_dimension = get_annotation_data_dimension(window_data)
	print(annotation_data_dimension)
		

	if model_type == 'neural_network':
		#global genomic_anno_to_gamma_model
		genomic_anno_to_gamma_model = init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'linear_model':
		#global genomic_anno_to_gamma_model
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'intercept_model':
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(1)
	#genomic_anno_to_gamma_model = None

	# VI Iterations
	for vi_iter in range(max_iterations):

		print('VI ITERATION ' + str(vi_iter))

		for subset_iter, window_data_subset in enumerate(np.array_split(window_data,30)):
			# Part 1 Update window_to_beta_mu and window_to_beta_mu given current values of window_to_gamma
			print('Data subset ' + str(subset_iter) + ' start')
			print(window_data_subset.shape)
			start_time = time.time()
			beta_squared_train, genome_anno_train = fast_update_beta_distributions(window_data_subset,genomic_anno_to_gamma_model,vi_iter, subset_iter, model_type)

			# Quick error check
			if np.sum(np.isnan(beta_squared_train)) > 0:
				print('ERROR: Nans in beta squared')
				pdb.set_trace()


			if model_type == 'intercept_model':
				genome_anno_train = np.ones((len(beta_squared_train), 1))

			# Part 2: Update annotations
			genomic_anno_to_gamma_model.train_on_batch(genome_anno_train, beta_squared_train)

			end_time = time.time()
			print(end_time-start_time)


		# Save model to output
		if np.mod(vi_iter, 5) == 0:
			genomic_anno_to_gamma_model.save(temp_output_model_root + '_' + str(vi_iter))

	return window_to_beta_mu, window_to_beta_var, window_to_gamma, genomic_anno_to_gamma_model

def debugging(genomic_anno_to_gamma_model, window_data):
	linear_params = np.asarray(genomic_anno_to_gamma_model.trainable_variables[0])[:,0]
	sldsc_res = np.loadtxt('/n/groups/price/ben/sldsc/results/blood_WHITE_COUNT_sldsc.results',dtype=str,delimiter='\t')
	sldsc_linear_params = sldsc_res[1:,-3].astype(float)
	# Get number of windows
	num_windows = window_data.shape[0]


	# Now loop through windows
	for window_iter in range(num_windows):
		# Extract relevent info for this window
		window_name = window_data['window_name'][window_iter]

		genomic_anno = np.load(window_data['genomic_annotation_file'][window_iter])
		
		mod_pred = genomic_anno_to_gamma_model.predict(genomic_anno)[:,0]

		pdb.set_trace()

def calculate_chi_sq_pred(model_pred_tau, ld, samp_size):
	ld_sq = np.square(ld)

	chi_sq_pred = []
	# loop through variants
	for ii in range(len(model_pred_tau)):
		pred = 1.0 + (samp_size*np.sum(ld_sq[ii,:]*model_pred_tau))
		chi_sq_pred.append(pred)
	return np.asarray(chi_sq_pred)

def calculate_ld_score_log_likelihood(chi_sq, chi_sq_pred, ld_scores):
	log_likes = gamma.pdf(chi_sq, .5, scale=2*chi_sq_pred)/ld_scores
	return np.asarray(log_likes)


def run_ld_score_regression_likelihood_evaluation(testing_window_data, genomic_anno_to_gamma_model, samp_size, output_file):
	# Get number of windows
	num_windows = testing_window_data.shape[0]

	t = open(output_file,'w')
	t.write('snp\twindow\tobserved_chi_sq\tpred_chi_sq\tldsc_gamma_log_like\tpred_tau\n')
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
		chi_sq = np.square(marg_beta/marg_beta_se)

		model_pred_tau = genomic_anno_to_gamma_model.predict(genomic_anno)[:,0]
		
		chi_sq_pred = calculate_chi_sq_pred(model_pred_tau, ld, samp_size)

		ld_scores = np.sum(np.square(ld),axis=0)

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
			t.write(variant_ids[ii] + '\t' + window_name + '\t' + str(chi_sq[ii]) + '\t' + str(chi_sq_pred[ii]) + '\t' + str(log_likelihoods[ii]) + '\t' + str(model_pred_tau[ii]) + '\n')
	t.close()


trait_name = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
model_type = sys.argv[4]
samp_size = float(sys.argv[5])

print(trait_name)
print(model_type)


# load in data
training_chromosome_type = 'chrom_1_18'
training_chromosome_type = 'even'
training_chromosomes = get_training_chromosomes(training_chromosome_type)
window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma = load_in_data(input_dir, trait_name, training_chromosomes)


# Model training
temp_output_model_root = output_dir + trait_name + '_nonlinear_sldsc_multivariate_updates_results_training_data_' + model_type + '_' + training_chromosome_type + '_annotations_to_gamma_model_temp'
window_to_beta_mu, window_to_beta_var, window_to_gamma, genomic_anno_to_gamma_model = non_linear_sldsc(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma, model_type, temp_output_model_root)


# Save training data results
training_data_snp_effects_output_file = output_dir + trait_name + '_nonlinear_sldsc_multivariate_updates_results_training_data_' + model_type + '_' + training_chromosome_type + '_training_snp_effects.txt'
save_snp_effects_on_training_data_to_output_file(window_data, window_to_beta_mu, window_to_beta_var, window_to_gamma, training_data_snp_effects_output_file)
# Save TensorFlow model
training_data_tf_model_output_file = output_dir + trait_name + '_nonlinear_sldsc_multivariate_updates_results_training_data_' + model_type + '_' + training_chromosome_type + '_annotations_to_gamma_model'
genomic_anno_to_gamma_model.save(training_data_tf_model_output_file)
#genomic_anno_to_gamma_model = tf.keras.models.load_model(training_data_tf_model_output_file,custom_objects={'gaussian_neg_log_likelihood_tf_padded_loss':gaussian_neg_log_likelihood_tf_padded_loss})





testing_chromosome_type = 'chrom_19_22'
testing_chromosome_type = 'odd'

testing_chromosomes = get_training_chromosomes(testing_chromosome_type)
testing_window_data = load_in_testing_data(input_dir, trait_name, testing_chromosomes)

testing_ldsc_likelihood_output_file = output_dir + trait_name + '_nonlinear_sldsc_multivariate_updates_results_training_data_' + model_type + '_' + training_chromosome_type + '_' + testing_chromosome_type + '_testing_ld_score_regression_eval.txt'

run_ld_score_regression_likelihood_evaluation(testing_window_data, genomic_anno_to_gamma_model, samp_size, testing_ldsc_likelihood_output_file)




#debugging(genomic_anno_to_gamma_model, window_data)

