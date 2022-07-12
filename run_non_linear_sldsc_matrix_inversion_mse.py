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
	elif training_chromosome_type == 'chrom_21':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 21:
				training_chromosomes[chrom_num] = 1
		return training_chromosomes
	elif training_chromosome_type == 'chrom_22':
		training_chromosomes = {}
		for chrom_num in range(1,23):
			if chrom_num == 22:
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



def window_includes_mhc_region(window_chrom_num, window_start, window_end, mhc_region):
	if window_chrom_num != '6':
		return False
	else:
		in_mhc = False
		for pos in range(window_start,window_end):
			if pos in mhc_region:
				in_mhc = True
		return in_mhc


def load_in_data(input_dir, trait_name, training_chromosomes, diagonal_padding_str):
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
	chi_sq_inv = []
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

			inv_chi_file = input_dir + trait_name + '_' + window_name + '_' + diagonal_padding_str + '_decorrelated_chi_squared.npy'
			chi_sq_inv.append(inv_chi_file)

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
	dd = {'window_name': window_names, 'variant_id_file': variant_id, 'srs_inv_file':srs_inv, 's_inv_2_diag_file':s_inv_2_diag, 'D_mat_file': D_mat, 'beta_file': beta, 'genomic_annotation_file':genomic_annotation, 'middle_variant_indices_file':middle_variant_indices, 'middle_hm3_variant_indices_file':middle_hm3_variant_indices, 'beta_se_file': beta_se, 'ld_file':ld, 'D_diag_file': D_diag, 'squared_ld_file': squared_ld, 'chi_sq_inv': chi_sq_inv}
	df = pd.DataFrame(data=dd)

	return df



def load_in_training_data_from_window_data(window_data):
	genome_anno_arr = []
	chi_sq_inv_arr = []
	# Total number of windows
	num_windows = window_data.shape[0]
	for window_iter in range(num_windows):
		window_name = window_data.iloc[window_iter]['window_name']
		window_middle_indices = np.load(window_data.iloc[window_iter]['middle_variant_indices_file'])
		genomic_anno = np.load(window_data.iloc[window_iter]['genomic_annotation_file'])
		chi_sq_inv = np.load(window_data.iloc[window_iter]['chi_sq_inv'])

		genome_anno_arr.append(genomic_anno[window_middle_indices, :])
		chi_sq_inv_arr.append(chi_sq_inv[window_middle_indices])
	genome_anno_arr = np.vstack(genome_anno_arr)
	chi_sq_inv_arr = np.hstack(chi_sq_inv_arr)
	return chi_sq_inv_arr, genome_anno_arr

def init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.Dense(units=64, activation='relu'))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))
	model.compile(loss='MeanSquaredError', optimizer='adam')
	return model

def init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=64, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))
	model.compile(loss='MeanSquaredError', optimizer='adam')
	return model




def init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=1, activation='softplus', input_dim=annotation_data_dimension))
	model.compile(loss='MeanSquaredError', optimizer='adam')
	return model

def init_exp_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=1, activation='exponential', input_dim=annotation_data_dimension))
	model.compile(loss='MeanSquaredError', optimizer='adam')

	return model

def get_annotation_data_dimension(window_data):
	genomic_anno_dim = np.load(window_data['genomic_annotation_file'][0]).shape[1]
	return genomic_anno_dim




def initialize_genomic_anno_model(model_type, annotation_data_dimension):
	if model_type == 'neural_network':
		genomic_anno_to_gamma_model = init_non_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'neural_network_no_drops':
		genomic_anno_to_gamma_model = init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'linear_model':
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'exp_linear_model':
		genomic_anno_to_gamma_model = init_exp_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'intercept_model':
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(1)

	return genomic_anno_to_gamma_model


def marginal_non_linear_sldsc(inv_chi_sq_train, genome_anno_train, model_type, max_epochs=200):
	# Number of annotations
	annotation_data_dimension = get_annotation_data_dimension(window_data)

	# Initialize mapping from annotations to per snp heritability
	genomic_anno_to_gamma_model = initialize_genomic_anno_model(model_type, annotation_data_dimension)

	# Train model
	genomic_anno_to_gamma_model.fit(genome_anno_train, inv_chi_sq_train, epochs=max_epochs)

	return genomic_anno_to_gamma_model

trait_name = sys.argv[1]
preprocessed_data_for_non_linear_sldsc_dir = sys.argv[2]
samp_size = int(sys.argv[3])
diagonal_padding_str = sys.argv[4]
non_linear_sldsc_results_dir = sys.argv[5]
model_type = sys.argv[6]



print(trait_name)
print(model_type)
print(diagonal_padding_str)


# load in data
training_chromosome_type = 'even'
training_chromosomes = get_training_chromosomes(training_chromosome_type)
window_data = load_in_data(preprocessed_data_for_non_linear_sldsc_dir, trait_name, training_chromosomes, diagonal_padding_str)

inv_chi_sq_train, genome_anno_train = load_in_training_data_from_window_data(window_data)


genomic_anno_to_gamma_model = marginal_non_linear_sldsc(inv_chi_sq_train, genome_anno_train, model_type, max_epochs=200)



# Save TensorFlow model
training_data_tf_model_output_file = non_linear_sldsc_results_dir + trait_name + '_nonlinear_sldsc_matrix_inversion_mse_updates_results_training_data_' + model_type + '_' + diagonal_padding_str + '_diagonal_padding_' + training_chromosome_type + '_annotations_to_gamma_model'
genomic_anno_to_gamma_model.save(training_data_tf_model_output_file)


