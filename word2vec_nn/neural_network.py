from __future__ import print_function
from sklearn.metrics import f1_score
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

def build_nn(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None, 80),
									 input_var=input_var)

	l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())


	l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=8,
            nonlinearity=lasagne.nonlinearities.softmax)

	return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(dataset, targets, model='mlp', num_epochs=500):
	print (dataset.ndim)
	input_var = T.matrix('datasetT')
	target_var = T.ivector('targetsT')

	network = build_nn(input_var)
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()

	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0025, momentum=0.9)
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	print("Starting training...")
    # We iterate over epochs:
	for epoch in range(num_epochs):
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(dataset, targets, 50, shuffle=True):
			inp, targ = batch
			train_err += train_fn(inp, targ)
			train_batches += 1
			print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


	net_output = lasagne.layers.get_output(network, deterministic=True)
	true_output = T.ivector('true_output')
	loss = T.mean(lasagne.objectives.categorical_crossentropy(net_output, true_output))
	get_output = theano.function([input_var], net_output)
	y_predicted = np.argmax(get_output(dataset), axis=1)
	print(100*np.mean(targets == y_predicted))

