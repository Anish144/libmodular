########################
#Variational Inference
##########################
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import distributions as tfd
from collections import namedtuple
import numpy as np
import pdb
import datetime
from tqdm import tqdm
from tensorflow.python import debug as tf_debug
import sys
import observations


REALRUN = sys.argv[1]


params = namedtuple('parameters', ['a', 'b', 'beta_prior', 
									'beta', 'selection', 'probs'])
ModulePool = namedtuple('ModulePool', ['module_fnc'])

layer_params = []


def generator(arrays, batch_size):
	"""Generate batches, one with respect to each array's first axis."""
	starts = [0] * len(arrays)  # pointers to where we are in iteration
	while True:
		batches = []
		for i, array in enumerate(arrays):
			start = starts[i]
			stop = start + batch_size
			diff = stop - array.shape[0]
			if diff <= 0:
				batch = array[start:stop]
				starts[i] += batch_size
			else:
				batch = np.concatenate((array[start:], array[:diff]))
				starts[i] = diff
			batches.append(batch)
		yield batches

def create_summary(list_of_ops_or_op, name, summary_type):
	summary = getattr(tf.summary, summary_type)

	if type(list_of_ops_or_op) is list:
		for i in range(len(list_of_ops_or_op)):
			summary(str(name) + '_' + str(i), list_of_ops_or_op[i])

	elif type(list_of_ops_or_op) is tf.Tensor:
		summary(str(name), list_of_ops_or_op)

	else:
		raise TypeError('Invalid type for summary')


def get_u(outputs):
	return tf.random_uniform(outputs, minval=0.0001, maxval=0.999)

def get_test_selection(a, b):
	denom = tf.exp(tf.lgamma(1 + tf.pow(a,-1) + b))
	term_1 = tf.exp(tf.lgamma(1 + tf.pow(a,-1)))
	term_2 = tf.multiply(b, tf.exp(tf.lgamma(b)))
	numerator = tf.multiply(term_1, term_2)
	return tf.divide(numerator, denom)

def give_input_dependent(standardised):
	zero = tf.constant(0.1, shape=[],
					  dtype=tf.float32)

	fill_zero = tf.where(standardised>zero,
				x=standardised,
				y=tf.zeros_like(standardised))
	fill_one = tf.where(standardised>zero,
						x=tf.ones_like(standardised),
						y=standardised)

	return fill_one

def selection_logprob():
	with tf.variable_scope('selection_logprob'):
		def layer_selection_logprob(layer):
			probs = layer_params[layer].probs
			selection = layer_params[layer].selection
			logprobs = tf.log(probs + 1e-20)
			log1probs = tf.log(1 - probs + 1e-20)
			sel = tf.cast(selection, tf.float32)
			term_1 = tf.multiply(logprobs, sel)
			term_2 = tf.multiply(log1probs, 1 - sel)
			return term_1 + term_2
		x = [tf.reduce_sum(layer_selection_logprob(layer), axis=-1) for layer in range(len(layer_params))]
		return tf.reduce_sum(x, axis=0)

def run_mask(inputs, outputs,  tau, rho, module: ModulePool, train):
	with tf.variable_scope(None, 'mask'):
		shape = outputs
		a = tf.get_variable(name='a', dtype=tf.float32, initializer=tf.random_uniform([shape], 
																				minval=0.1, maxval=0.5))
		b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.random_uniform([shape], 
																				minval=0.2, maxval=0.2))

		u = get_u([tf.shape(inputs)[0], shape])
		div_a  = tf.divide(1 + 1e-10, a + 1e-20)
		div_b = tf.divide(1 + 1e-10, b + 1e-20)
		pow_ub = u**div_b
		pi = (1 - pow_ub)**div_a


		dep_shape = inputs.shape[-1].value
		eta = tf.get_variable(name='eta', shape=[dep_shape], dtype=tf.float32)
		khi = tf.get_variable(name='khi', shape=[dep_shape], dtype=tf.float32)
		beta = tfd.MultivariateNormalDiag(eta, khi**2)

		gamma = tf.get_variable(name='gamma', shape=[dep_shape], dtype=tf.float32)


		beta_prior = tfd.MultivariateNormalDiag(tf.zeros([dep_shape]), 
		tf.multiply(rho, tf.ones([dep_shape]))
											)

		term_1 = tf.log(tf.divide(pi, 1-pi + 1e-20) + 1e-20)
		term_2 = tf.log(tf.divide(u, 1-u + 1e-20) + 1e-20)
		z = tf.sigmoid(tf.multiply(tf.divide(1,tau), term_1 + term_2))

		test_pi = get_test_selection(a, b)
		test_selection = tf.where(test_pi>0.5,
								x=tf.ones_like(test_pi),
								y=tf.zeros_like(test_pi)) 
		test_selection = tf.tile(test_selection, [tf.shape(inputs)[0]])
		test_selection = tf.reshape(test_selection,[shape, tf.shape(inputs)[0]])
		test_selection = tf.transpose(test_selection, [1,0])

		g = tf.get_default_graph()
		with g.gradient_override_map({"Round": "Identity"}):
			selection = tf.cond(train, 
			lambda: tf.round(z),
			lambda: test_selection
			)

		p = params(a, b, beta_prior, beta, z, pi)
		layer_params.append(p)
		return run_modules(inputs, selection, module.module_fnc), a, b, selection, pi

def run_modules(inputs, z, module_fnc):
	return module_fnc(inputs, z)


def get_weights(in_shape, out_shape, activation=None):
	with tf.variable_scope(None, 'layer'):
		W = tf.get_variable(name='weights', shape=[in_shape, out_shape])
		bias = tf.get_variable(name='bias', shape=[out_shape])

		def module_fnc(x, z):
			g = tf.get_default_graph()
			masked_W = tf.einsum('no,ko->nko', z, W)
			out = tf.einsum('nk,nko->no', x, masked_W) + bias
			if activation is not None:
				out = activation(out)
			return out

	return ModulePool(module_fnc)

def get_variational_kl(alpha):

	def get_layer_KL(number):
		a = layer_params[number].a
		b = layer_params[number].b
		term_1 = tf.divide(- b + 1, b)
		term_2 = tf.log( tf.divide(tf.multiply(a, b), alpha) + 1e-20)
		term_bracket = (tf.digamma(1.) - tf.digamma(b) - tf.divide(1., b + 1e-20))
		term_3 = tf.multiply(tf.divide(a - alpha, a + 1e-20), term_bracket)
		return tf.reduce_sum(term_1 + term_2 + term_3)

	return tf.reduce_sum([get_layer_KL(i) for i in range(len(layer_params))])

def run():
	(x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')
	dataset_size = x_train.shape[0]

	#Data
	data = tf.placeholder(tf.float32, [None, 784])
	target = tf.placeholder(tf.int32, [None])
	istraining = tf.placeholder(shape=[], dtype=tf.bool)

	#Network
	data_input = tf.layers.dense(data, 640, tf.nn.relu)

	units = 64
	Module = get_weights(640, units, tf.nn.relu)
	logits, a1, b1, sel1, pi1 = run_mask(data_input, units, 0.1, 1., Module, istraining)

	sel_1 = tf.divide(tf.reduce_mean(tf.reduce_sum(sel1, axis=1)), units)

	units = 32
	Module = get_weights(64, units, tf.nn.relu)
	logits, a2, b2, sel2, pi2 = run_mask(logits, units, 0.1, 1., Module, istraining)

	sel_2 = tf.divide(tf.reduce_mean(tf.reduce_sum(sel2, axis=1)), units)

	logits = tf.layers.dense(logits, 10)

	#Loss
	loglikelihood = tf.distributions.Categorical(logits).log_prob(target)
	alpha = 0.1
	KL = get_variational_kl(alpha)
	selection_prob = selection_logprob()

	loglike = 60000 * tf.reduce_mean(loglikelihood + selection_prob)

	#Reduce_mean the likelihood across the batches as an estimator
	elbo = (loglike - KL)

	optimizer = tf.train.AdamOptimizer()
	opt = optimizer.minimize(-elbo)

	#Accuracy metric
	activate = tf.nn.softmax(logits)
	predicted = tf.argmax(activate, axis=1, output_type=tf.int32)
	acc = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

	# create_summary(acc, 'Accuracy', 'scalar')
	# create_summary(elbo, 'ELBO', 'scalar')
	# create_summary(KL, 'KL', 'scalar')
	# create_summary(loglike, 'LogLike', 'scalar')

	# create_summary([a1, a2], 'a', 'histogram')
	# create_summary([b1, b2], 'b', 'histogram')

	with tf.Session() as sess:
		# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
		sess.run(init)
		general_summaries = tf.summary.merge_all()

		if REALRUN == 'True':
			test_writer = tf.summary.FileWriter(
				f'logs/test:Cifar10_Variationl_with_straightthrough_estimator_and_selectionlogprob_{time}', sess.graph)
			writer = tf.summary.FileWriter(
				f'logs/train:Cifar10_Variationl_with_straightthrough_estimator_and_selectionlogprob_{time}', sess.graph)

		batches = generator([x_train, y_train, np.arange(dataset_size)], 64)
		for i, (batch_x, batch_y, indices) in tqdm(enumerate(batches)):
			print('HELLO')
			# if i % 100 == 0:
			# 	xs = x_test
			# 	ys = y_test
			# 	test_dict = {data: xs, target: ys, istraining:False}
			# 	summary = sess.run(general_summaries, feed_dict = test_dict)
			# 	test_writer.add_summary(summary, global_step=i)


			train_dict = {data:x_train,  target:y_train, istraining:True}

			_  = sess.run(opt, train_dict)

if __name__ == '__main__':
	run()











