import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas as pd
import os
import pdb
import tensorflow as tf
import gzip
import time



def simple_simulation(N=500000):
	anno = np.ones((N, 2))
	anno[:50000] = anno[:50000]*0.0


	beta_squared = []
	true_varz = []

	for nn in range(N):
		if anno[nn,1] == 0.0:
			var = 3.0
		elif anno[nn,1] == 1.0:
			var = 0.8
		true_varz.append(var)
		beta = np.random.normal(loc=0.0, scale=np.sqrt(var))
		beta_squared.append(np.square(beta))
	return np.asarray(beta_squared), anno, np.asarray(true_varz)


def gaussian_neg_log_likelihood_tf_loss(beta_squared_true, gamma_pred):
	nll = tf.math.log(gamma_pred) + tf.math.divide(beta_squared_true, gamma_pred)
	return nll

def create_non_linear_mapping_from_genomic_annotations_to_gamma(beta_squared_train, genome_anno_train):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=128, activation='relu', input_dim=genome_anno_train.shape[1]))
	model.add(tf.keras.layers.Dense(units=128, activation='relu'))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	model.compile(loss=gaussian_neg_log_likelihood_tf_loss, optimizer='adam')

	model.fit(genome_anno_train, beta_squared_train, epochs=10)

	return model











beta_squared, annos, true_varz = simple_simulation()


non_linear_mapping_fxn = create_non_linear_mapping_from_genomic_annotations_to_gamma(beta_squared, annos)

print(non_linear_mapping_fxn.predict(annos)[:,0])

pdb.set_trace()