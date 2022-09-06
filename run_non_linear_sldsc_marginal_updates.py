import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas as pd
import os
import pdb
import tensorflow as tf
import tensorflow_recommenders as tfrs
import gzip
import time
from numba import njit, prange, jit
from joblib import Parallel, delayed
from scipy.stats import gamma
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects


def get_training_and_evaluation_chromosomes(training_chromosome_type, evaluation_chromosome_type):
	if evaluation_chromosome_type == 'chr_14':
		evaluation_chromosomes = {}
		evaluation_chromosomes[14] = 1
	if evaluation_chromosome_type == 'chr_15':
		evaluation_chromosomes = {}
		evaluation_chromosomes[15] = 1


	if training_chromosome_type == 'even':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if np.mod(chrom_num,2) == 0:
				if chrom_num not in evaluation_chromosomes:
					training_chromosomes[chrom_num] = 1
	if training_chromosome_type == 'odd':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if np.mod(chrom_num,2) == 1:
				if chrom_num not in evaluation_chromosomes:
					training_chromosomes[chrom_num] = 1
	if training_chromosome_type == 'all':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num not in evaluation_chromosomes:
				training_chromosomes[chrom_num] = 1

	return training_chromosomes	, evaluation_chromosomes	

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


def load_in_data(input_dir, trait_name, ld_type, training_chromosomes):
	window_names = []
	variant_id = []
	beta = []
	beta_se = []
	squared_ld = []
	regression_snps_squared_ld = []
	genomic_annotation = []
	middle_variant_indices = []
	regression_variant_indices = []
	middle_regression_variant_indices = []



	for chrom_num in range(1,23):
		if chrom_num not in training_chromosomes:
			continue
		input_file = input_dir + 'genome_wide_susie_windows_and_non_linear_sldsc_processed_data_' + ld_type + '_chrom_' + str(chrom_num) + '.txt'
		head_count = 0
		f = open(input_file)
		for line in f:
			line = line.rstrip()
			data = line.split('\t')
			if head_count == 0:
				head_count = head_count + 1
				continue
			# Extract revlevent fields from line
			window_name = data[0]

			window_chrom_num = window_name.split(':')[0]
			window_start = int(window_name.split(':')[1])
			window_end = int(window_name.split(':')[2])

			window_names.append(data[0])
			variant_id.append(data[1])
			beta.append(data[2] + trait_name + '.npy')
			beta_se.append(data[3] + trait_name + '.npy')
			genomic_annotation.append(data[4])
			middle_variant_indices.append(data[5])
			regression_variant_indices.append(data[6])
			middle_regression_variant_indices.append(data[7])
			squared_ld.append(data[8])
			regression_snps_squared_ld.append(data[9])
		f.close()
	# Quick error checking
	if len(np.unique(window_names)) != len(window_names):
		print('assumption error')
		pdb.set_trace()
	# Put data in pandas df
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices, 'regression_variant_indices_file':regression_variant_indices, 'middle_regression_variant_indices_file': middle_regression_variant_indices, 'beta_se_file': beta_se, 'squared_ld_file': squared_ld, 'regression_snps_squared_ld_file':regression_snps_squared_ld}
	df = pd.DataFrame(data=dd)
	return df

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
	if weights == None:
		gamma = np.ones(len(window_marginal_betas))*1e-5
	else:
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




def update_beta_distribution_in_single_window_univariate(window_name, window_srs_inv_file, window_s_inv_2_diag_file, window_D_diag_file, window_marginal_betas_file, beta_mu, beta_var, weights, model_type, window_genomic_annotation_file, window_middle_variant_indices_file):
	window_srs_inv = np.load(window_srs_inv_file)
	window_s_inv_2_diag = np.load(window_s_inv_2_diag_file)
	window_D_diag = np.load(window_D_diag_file)
	window_marginal_betas = np.load(window_marginal_betas_file)
	genomic_anno = np.load(window_genomic_annotation_file)
	# UPDATES for this window
	num_snps = len(window_marginal_betas)

	middle_indices = np.load(window_middle_variant_indices_file)


	#diff=genomic_anno_to_gamma_model.predict(genomic_anno) - tf.math.softplus(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(genomic_anno, weights[0]) + weights[1]), weights[2]) + weights[3]), weights[4]) + weights[5])
	if weights == None:
		gamma = np.ones(len(window_marginal_betas))*1e-5
	else:
		if model_type == 'neural_network' or model_type == 'neural_network_no_drops':
			gamma = softplus_np(np.dot(relu_np(np.dot(relu_np(np.dot(genomic_anno, weights[0]) + weights[1]), weights[2]) + weights[3]), weights[4]) + weights[5])
			gamma = gamma[:,0]
		elif model_type == 'linear_model':
			gamma = softplus_np(np.dot(genomic_anno, weights[0]) + weights[1])
			gamma = gamma[:,0]
		elif model_type == 'intercept_model':
			genomic_anno = np.ones((genomic_anno.shape[0],1))
			gamma = softplus_np(np.dot(genomic_anno, weights[0]) + weights[1])
			gamma = gamma[:,0]

	# Marginal betas will all effects removed
	residual = window_marginal_betas - np.dot(window_srs_inv, beta_mu)

	# Loop through snps
	for vi_iter in range(40):
		for k_index in range(num_snps):

			# get marginal betas with all effects removed other than the snp of interest
			residual = residual + window_srs_inv[:, k_index]*beta_mu[k_index]

			# Calculate terms involved in the update
			b_term = residual[k_index]*window_s_inv_2_diag[k_index]
			a_term = (-.5*window_D_diag[k_index]) - (.5/gamma[k_index])

			# VI Updates
			beta_var[k_index] = -1.0/(2.0*a_term)
			beta_mu[k_index] = b_term*beta_var[k_index]

			# Update resid for next round (after this resid includes effects of all genes)
			residual = residual - window_srs_inv[:,k_index]*beta_mu[k_index]

	genomic_anno_train = genomic_anno[middle_indices,:]
	return (beta_mu, beta_var, genomic_anno_train, middle_indices)


def fast_update_beta_distributions(window_data, genomic_anno_to_gamma_model, vi_iter, subset_iter, model_type, window_to_beta_mu, window_to_beta_var, parallel_bool=True):
	# Get number of windows
	num_windows = window_data.shape[0]

	if genomic_anno_to_gamma_model == None:
		weights = None
	else:
		weights = genomic_anno_to_gamma_model.get_weights()


	if parallel_bool == False:
		beta_data = []
		# Now loop through windows
		for window_iter in range(num_windows):
			print(window_iter)
			beta_data.append(update_beta_distribution_in_single_window_univariate(window_data.iloc[window_iter]['window_name'], window_data.iloc[window_iter]['srs_inv_file'], window_data.iloc[window_iter]['s_inv_2_diag_file'], window_data.iloc[window_iter]['D_diag_file'], window_data.iloc[window_iter]['beta_file'], window_to_beta_mu[window_data.iloc[window_iter]['window_name']], window_to_beta_var[window_data.iloc[window_iter]['window_name']], weights, model_type, window_data.iloc[window_iter]['genomic_annotation_file'], window_data.iloc[window_iter]['middle_variant_indices_file']))
	elif parallel_bool == True:
		beta_data = Parallel(n_jobs=20)(delayed(update_beta_distribution_in_single_window_univariate)(window_data.iloc[window_iter]['window_name'], window_data.iloc[window_iter]['srs_inv_file'], window_data.iloc[window_iter]['s_inv_2_diag_file'], window_data.iloc[window_iter]['D_diag_file'], window_data.iloc[window_iter]['beta_file'], window_to_beta_mu[window_data.iloc[window_iter]['window_name']], window_to_beta_var[window_data.iloc[window_iter]['window_name']], weights, model_type, window_data.iloc[window_iter]['genomic_annotation_file'], window_data.iloc[window_iter]['middle_variant_indices_file']) for window_iter in range(num_windows))


	# GET ORGANIZED NEURAL NET TRAINING DATA
	beta_squared_train = []
	genomic_anno_train = []
	for window_iter in range(num_windows):
		# Extract relevent info for this window
		window_name = window_data.iloc[window_iter]['window_name']
		temp_beta_mu = beta_data[window_iter][0]
		temp_beta_var = beta_data[window_iter][1]
		temp_beta_squared = np.square(temp_beta_mu) + temp_beta_var

		middle_indices = beta_data[window_iter][3]
		beta_squared_train.append(temp_beta_squared[middle_indices])
		genomic_anno_train.append(beta_data[window_iter][2])

		window_to_beta_mu[window_name] = temp_beta_mu
		window_to_beta_var[window_name] = temp_beta_var

	return np.hstack(beta_squared_train), np.vstack(genomic_anno_train), window_to_beta_mu, window_to_beta_var



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


def gaussian_neg_log_likelihood_np_padded_loss(beta_squared_true, gamma_pred):
	epsilon = 1e-30
	nll = np.log(gamma_pred + epsilon) + np.divide(beta_squared_true, (gamma_pred+epsilon))
	return nll

def init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	return model

def init_non_linear_batch_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	return model


def init_non_linear_layer_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	return model

def init_non_linear_layer_norm_experiment_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	return model

def init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, dropout_rate):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dropout(dropout_rate))
	model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.Dropout(dropout_rate))
	model.add(tf.keras.layers.Dense(units=64, activation='relu'))
	model.add(tf.keras.layers.Dropout(dropout_rate))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	return model

def init_reduced_dimension_interaction_from_genomic_annotations_to_gamma(annotation_data_dimension, latent_space_size=5):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=latent_space_size, activation='relu', input_dim=annotation_data_dimension))
	model.add(tfrs.layers.dcn.Cross())
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))
	return model

def init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=1, activation='softplus', input_dim=annotation_data_dimension))

	return model

def init_exp_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=1, activation='exponential', input_dim=annotation_data_dimension))

	return model

def get_annotation_data_dimension(window_data):
	genomic_anno_dim = np.load(window_data['genomic_annotation_file'][0]).shape[1]
	return genomic_anno_dim




def initialize_genomic_anno_model(model_type, annotation_data_dimension):
	if model_type == 'neural_network_10':
		genomic_anno_to_gamma_model = init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, .10)
	elif model_type == 'neural_network_20':
		genomic_anno_to_gamma_model = init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, .20)
	elif model_type == 'neural_network_30':
		genomic_anno_to_gamma_model = init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, .30)
	elif model_type == 'neural_network_40':
		genomic_anno_to_gamma_model = init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, .40)
	elif model_type == 'neural_network_no_drops':
		genomic_anno_to_gamma_model = init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, False)
	elif model_type == 'neural_network_no_drops_scale':
		genomic_anno_to_gamma_model = init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	elif model_type == 'neural_network_batch_norm':
		genomic_anno_to_gamma_model = init_non_linear_batch_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	elif model_type == 'neural_network_layer_norm':
		genomic_anno_to_gamma_model = init_non_linear_layer_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	elif model_type == 'neural_network_layer_norm_experiment':
		genomic_anno_to_gamma_model = init_non_linear_layer_norm_experiment_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	elif model_type == 'linear_model':
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'exp_linear_model':
		genomic_anno_to_gamma_model = init_exp_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'intercept_model':
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(1)
	elif model_type == 'reduced_dimension_interaction_model':
		genomic_anno_to_gamma_model = init_reduced_dimension_interaction_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'reduced_dimension_neural_network_model':
		genomic_anno_to_gamma_model = init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 5, False)
	elif model_type == 'reduced_dimension_neural_network_model_scale':
		genomic_anno_to_gamma_model = init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 5, True)
	return genomic_anno_to_gamma_model


def ldsc_tf_loss_fxn(chi_sq, pred_tau, samp_size, ld_sq, snp_weights, intercept_variable):
	pred_chi_sq = (samp_size*tf.linalg.matmul(ld_sq, pred_tau)) + (tf.math.softplus(intercept_variable))

	log_like = (-.5)*tf.math.log(chi_sq) - tf.math.divide(chi_sq, 2.0*pred_chi_sq) - (.5*tf.math.log(2.0*pred_chi_sq))

	weighted_middle_indices_log_like = tf.math.divide(log_like, snp_weights)

	return -tf.math.reduce_sum(weighted_middle_indices_log_like), log_like

def filter_window_middle_indices_to_independent_snps_at_random(squared_ld, window_middle_indices, thresh=.0025):
	num_snps = squared_ld.shape[0]
	all_snps = np.arange(num_snps)
	valid_snps = np.asarray([True]*num_snps)
	randomly_selected_independent_snps = []
	keep_going = True
	

	while keep_going:

		snp_name = np.random.choice(all_snps[valid_snps])
		randomly_selected_independent_snps.append(snp_name)

		new_valid_snps = squared_ld[snp_name,:] <= thresh
		valid_snps = valid_snps*new_valid_snps
		if np.sum(valid_snps) == 0:
			keep_going = False

	randomly_selected_independent_snps = np.asarray(randomly_selected_independent_snps)
	return window_middle_indices[randomly_selected_independent_snps]

def check_symmetric(a, rtol=1e-05, atol=1e-08):
	return numpy.allclose(a, a.T, rtol=rtol, atol=atol)


def calculate_loss_on_evaluation_data(evaluation_window_data, genomic_anno_to_gamma_model, log_intercept_variable):
	num_windows = evaluation_window_data.shape[0]
	epoch_eval_log_likelihoods = []
	epoch_eval_weights = []
	for window_counter, window_iter in enumerate(range(num_windows)):
		# Load in data for this window
		window_name = evaluation_window_data.iloc[window_iter]['window_name']

		# Extract chi-squared statistics
		window_beta = np.load(evaluation_window_data.iloc[window_iter]['beta_file'])
		window_beta_se = np.load(evaluation_window_data.iloc[window_iter]['beta_se_file'])
		window_chi_sq = np.square(window_beta/window_beta_se)
		# Extract indices in the middle of window as well as regression snps
		regression_indices = np.load(evaluation_window_data.iloc[window_iter]['regression_variant_indices_file'])
		window_middle_indices = np.load(evaluation_window_data.iloc[window_iter]['middle_regression_variant_indices_file'])
			
		# Extract genomic annotation file
		window_genomic_anno = np.load(evaluation_window_data.iloc[window_iter]['genomic_annotation_file'])
		squared_ld = np.load(evaluation_window_data.iloc[window_iter]['regression_snps_squared_ld_file'])

		# Weight middle-regression snps by all regression snps
		snp_weights = np.sum(squared_ld[:, regression_indices], axis=1)
		snp_weights[snp_weights < 1.0]=1.0

		# Convert to tensors
		snp_weights = snp_weights.reshape(len(snp_weights),1)
		window_chi_sq = window_chi_sq[window_middle_indices]
		window_chi_sq = tf.convert_to_tensor(window_chi_sq.reshape(len(window_chi_sq),1), dtype=tf.float32)
		squared_ld = tf.convert_to_tensor(squared_ld, dtype=tf.float32)


		# If using intercept, alter genomic annotations
		if model_type == 'intercept_model':
			window_genomic_anno = np.ones((len(window_beta), 1))

		window_pred_tau = genomic_anno_to_gamma_model(window_genomic_anno, training=False)

		loss_value, log_likelihoods = ldsc_tf_loss_fxn(window_chi_sq, window_pred_tau, samp_size, squared_ld, snp_weights, log_intercept_variable)

		# Add eval log likelihoods and weights to array to keep track
		epoch_eval_log_likelihoods.append(np.asarray(log_likelihoods[:,0]))
		epoch_eval_weights.append(1.0/np.asarray(snp_weights[:,0]))

	epoch_eval_log_likelihoods = np.hstack(epoch_eval_log_likelihoods)
	epoch_eval_weights = np.hstack(epoch_eval_weights)
	epoch_eval_loss = np.sum(-epoch_eval_log_likelihoods*epoch_eval_weights)/np.sum(epoch_eval_weights)


	return epoch_eval_loss, genomic_anno_to_gamma_model, log_intercept_variable


def marginal_non_linear_sldsc(window_data, evaluation_window_data, samp_size, model_type, learn_intercept, temp_output_model_root, learning_rate, max_epochs=200):
	# Number of annotations
	annotation_data_dimension = get_annotation_data_dimension(window_data)

	# Initialize mapping from annotations to per snp heritability
	genomic_anno_to_gamma_model = initialize_genomic_anno_model(model_type, annotation_data_dimension)
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	
	# Whether or not to learn intercept in LDSC
	# Initial value is np.log(np.exp(1)-1.0) [which equals 1 when put through softplus activation function]
	if learn_intercept == 'learn_intercept':
		log_intercept_variable = tf.Variable(initial_value=0.541324854612918,trainable=True, name='intercept')
	elif learn_intercept == 'fixed_intercept':
		log_intercept_variable = tf.Variable(initial_value=0.541324854612918,trainable=False, name='intercept')
	else:
		print('assumption error: intercept model called ' + learn_intercept + ' not currently implemented')
		pdb.set_trace()

	# Initialize vectors to keep track of training and evaluation loss
	training_loss = []
	evaluation_loss = []

	# A window is a region of dna space
	# This is number of windows we split dna space
	num_windows = window_data.shape[0]

	# Lopp through windows
	for epoch_iter in range(max_epochs):
		# Keep track of training log likelihoods and weights of each regression snp
		#epoch_training_log_likelihoods = []
		#epoch_training_weights = []

		# Loop through windows
		print('###################################')
		print('epoch iter ' + str(epoch_iter))
		print('###################################')
		start_time = time.time()
		for window_counter, window_iter in enumerate(np.random.permutation(range(num_windows))):
			print(window_counter)
			# Load in data for this window
			window_name = window_data.iloc[window_iter]['window_name']

			# Extract chi-squared statistics
			window_beta = np.load(window_data.iloc[window_iter]['beta_file'])
			window_beta_se = np.load(window_data.iloc[window_iter]['beta_se_file'])
			window_chi_sq = np.square(window_beta/window_beta_se)
			# Extract indices in the middle of window as well as regression snps
			regression_indices = np.load(window_data.iloc[window_iter]['regression_variant_indices_file'])
			window_middle_indices = np.load(window_data.iloc[window_iter]['middle_regression_variant_indices_file'])
			
			# Extract genomic annotation file
			window_genomic_anno = np.load(window_data.iloc[window_iter]['genomic_annotation_file'])
			squared_ld = np.load(window_data.iloc[window_iter]['regression_snps_squared_ld_file'])

			# Weight middle-regression snps by all regression snps
			snp_weights = np.sum(squared_ld[:, regression_indices], axis=1)
			snp_weights[snp_weights < 1.0]=1.0


			# Convert to tensors
			snp_weights = snp_weights.reshape(len(snp_weights),1)
			window_chi_sq = window_chi_sq[window_middle_indices]
			window_chi_sq = tf.convert_to_tensor(window_chi_sq.reshape(len(window_chi_sq),1), dtype=tf.float32)
			squared_ld = tf.convert_to_tensor(squared_ld, dtype=tf.float32)


			# If using intercept, alter genomic annotations
			if model_type == 'intercept_model':
				window_genomic_anno = np.ones((len(window_beta), 1))

			# Use tf.gradient tape to compute gradients
			with tf.GradientTape() as tape:
				window_pred_tau = genomic_anno_to_gamma_model(window_genomic_anno, training=True)
				loss_value, log_likelihoods = ldsc_tf_loss_fxn(window_chi_sq, window_pred_tau, samp_size, squared_ld, snp_weights, log_intercept_variable)
			
			# Define trainable variables
			trainable_variables = genomic_anno_to_gamma_model.trainable_weights
			if learn_intercept == 'learn_intercept':
				trainable_variables.append(log_intercept_variable)
			# Compute and apply gradients
			grads = tape.gradient(loss_value, trainable_variables)
			optimizer.apply_gradients(zip(grads, trainable_variables))

		end_time = time.time()
		print('iteration run time: ' + str(end_time-start_time))

		# At the end of each epoch do some stuff
		#if np.mod(epoch_iter, 5) == 0:
		if True:
			# Save model data
			genomic_anno_to_gamma_model.save(temp_output_model_root + '_' + str(epoch_iter))
			np.save(temp_output_model_root + '_intercept_variable_' + str(epoch_iter) + '.npy', np.asarray(tf.math.softplus(log_intercept_variable)))

			# Compute weighted average evaluation loss
			epoch_evaluation_loss, genomic_anno_to_gamma_model, log_intercept_variable = calculate_loss_on_evaluation_data(evaluation_window_data, genomic_anno_to_gamma_model, log_intercept_variable)
			epoch_training_loss, genomic_anno_to_gamma_model, log_intercept_variable = calculate_loss_on_evaluation_data(window_data, genomic_anno_to_gamma_model, log_intercept_variable)
			evaluation_loss.append(epoch_evaluation_loss)
			training_loss.append(epoch_training_loss)

			# Print training and evaluation loss
			print('Training loss: ' + str(epoch_training_loss))
			print('Evaluation loss: ' + str(epoch_evaluation_loss))

			# Save training and evaluation losses to output file
			np.savetxt(temp_output_model_root + '_' + str(epoch_iter) + '_training_loss.txt', np.asarray(training_loss).astype(str), fmt="%s",delimiter='\t')
			np.savetxt(temp_output_model_root + '_' + str(epoch_iter) + '_evaluation_loss.txt', np.asarray(evaluation_loss).astype(str), fmt="%s",delimiter='\t')

			# Print model parameters
			intercept_variable = np.asarray(tf.math.softplus(log_intercept_variable))*1.0
			print('Intercept: ' + str(intercept_variable))
			print('Sorted tau subset:')
			print(np.sort(np.asarray(window_pred_tau[:,0])))

	return genomic_anno_to_gamma_model

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
training_chromosome_type = sys.argv[6]
evaluation_chromosome_type = sys.argv[7]
ld_type = sys.argv[8]
learn_intercept = sys.argv[9]
learning_rate_str = sys.argv[10]




max_epochs = 2000

print(trait_name)
print(model_type)
print(training_chromosome_type)
print(evaluation_chromosome_type)
print(ld_type)
print(learn_intercept)

# load in data
training_chromosomes, evaluation_chromosomes = get_training_and_evaluation_chromosomes(training_chromosome_type, evaluation_chromosome_type)
window_data = load_in_data(input_dir, trait_name, ld_type, training_chromosomes)
evaluation_window_data = load_in_data(input_dir, trait_name, ld_type, evaluation_chromosomes)

# Output root stem
model_output_root = output_dir + trait_name + '_nonlinear_sldsc_marginal_updates_gamma_likelihood_results_training_data_' + model_type + '_' + learn_intercept + '_' + ld_type + '_' + learning_rate_str + '_' + training_chromosome_type + '_train_' + evaluation_chromosome_type + '_eval' + '_annotations_to_gamma_model'

# Model training
genomic_anno_to_gamma_model = marginal_non_linear_sldsc(window_data, evaluation_window_data, samp_size, model_type, learn_intercept, model_output_root, float(learning_rate_str), max_epochs=max_epochs)


# Save TensorFlow model
genomic_anno_to_gamma_model.save(model_output_root)

