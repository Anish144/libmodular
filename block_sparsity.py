from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import distributions as tfd
from collections import namedtuple
import numpy as np
import pdb
from tqdm import tqdm
import observations
import datetime

from blocksparse.matmul import BlocksparseMatMul

tfd = tf.contrib.distributions

BlockPool = namedtuple('BlockPool', ['units', 'block_r', 'block_c'])
Parameters = namedtuple('Parameters', ['a', 'b'])
layers = []

def create_summary(list_of_ops_or_op, name, summary_type):
    summary = getattr(tf.summary, summary_type)

    if type(list_of_ops_or_op) is list:
        for i in range(len(list_of_ops_or_op)):
            summary(str(name) + '_' + str(i), list_of_ops_or_op[i])

    elif type(list_of_ops_or_op) is tf.Tensor:
        summary(str(name), list_of_ops_or_op)

    else:
        raise TypeError('Invalid type for summary')

def generator(arrays, batch_size):
	"""Generate batches, one with respect to each array's first axis."""
	starts = [0] * len(arrays)
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


def run_weights(data, units, block_size, activation=None):
	with tf.variable_scope(None, 'layer'):

		in_shape = data.shape[-1].value

		block = BlockPool(units, in_shape//block_size, units//block_size)
		sparse_pattern = generate_sparse_pattern(block)

		bsmm = BlocksparseMatMul(sparse_pattern, block_size=block_size)
		W = tf.get_variable(name='weights', shape=[in_shape, units])
		bias = tf.get_variable(name='bias', shape=[units])

		def module_fnc(x):
			output = bsmm(x, W) + bias
			if activation is not None:
				activation(output)
			return output

	return module_fnc(data), sparse_pattern

def generate_sparse_pattern(block: BlockPool):
	"""
	Non input dependent block sparse masks based on IBP
	"""
	with tf.variable_scope(None, 'Block'):
		block_shape = block.block_r * block.block_c
		a = tf.get_variable(name='a', initializer=tf.random_uniform([block_shape], 
																	minval=10, maxval=30))
		b = tf.get_variable(name='b', initializer=tf.random_uniform([block_shape], 
																	minval=10, maxval=30))
		u = get_u(block_shape)
		layers.append(Parameters(a, b))

		pi = tf.pow((1 - tf.pow(u, tf.divide(1,b + 1e-20))), tf.divide(1,a + 1e-20))

		z = get_relaxed_bernoulli(pi, 0.01, u)

		return tf.reshape(z, [block.block_r, block.block_c])

def get_relaxed_bernoulli(pi, tau, u):
	term_1 = tf.log(tf.divide(pi, 1-pi))
	term_2 = tf.log(tf.divide(u, 1-u))
	return tf.sigmoid(tf.multiply(tf.divide(1, tau), term_1 + term_2))

def get_u(shape):
	return tf.random_uniform([shape], maxval=1)

def get_variational_kl(alpha):
	def get_layer_KL(number):
		a = layers[number].a
		b = layers[number].b
		term_1 = tf.divide(- b + 1, b)
		term_2 = tf.log( tf.divide(tf.multiply(a, b), alpha))
		term_bracket = (tf.digamma(1.) - tf.digamma(b) - tf.divide(1., b))
		term_3 = tf.multiply(tf.divide(a - alpha, a), term_bracket)
		return tf.reduce_sum(term_1 + term_2 + term_3)
	return tf.reduce_mean([get_layer_KL(i) for i in range(len(layers))])

def run():

	#Load data
	(x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')
	dataset_size = x_train.shape[0]

	inputs = tf.placeholder(tf.float32, [None, 28 * 28], 'inputs')
	labels = tf.placeholder(tf.int32, [None], 'labels')

	sp_hidd_log = []
	sp_logit_log = []

	def network():

		hidden, sparse_hidden = run_weights(inputs, units=64, block_size=8, 
											activation=tf.nn.relu)
		hidden, sparse_logits = run_weights(hidden, units=32, block_size=8)

		logits = tf.layers.dense(hidden, 10)

		loglikelihood = tf.distributions.Categorical(logits).log_prob(labels)

		predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), tf.float32))

		sp_hidd_log.append(
			tf.reshape(sparse_hidden, 
				[1, tf.shape(sparse_hidden)[0], 
				tf.shape(sparse_hidden)[1], 1]
				))
		sp_logit_log.append(
			tf.reshape(sparse_logits, 
				[1, tf.shape(sparse_logits)[0], 
				tf.shape(sparse_logits)[1], 1]
				))

		return (tf.reduce_mean(loglikelihood), accuracy, 
				sparse_hidden, sparse_logits)

	loglike, acc, sp_hid, sp_log = network()

	KL = get_variational_kl(0.3)

	optimizer = tf.train.AdamOptimizer()
	opt = optimizer.minimize(-loglike+KL)

	#Generate Summaries
	create_summary(sp_hidd_log, 'Sparse Pattern 1', 'image')
	create_summary(sp_logit_log, 'Sparse Pattern 2', 'image')
	create_summary(loglike, 'loglikelihood', 'scalar')
	create_summary(acc, 'accuracy', 'scalar')


	with tf.Session() as sess:
		init = tf.group(tf.global_variables_initializer(), 
			tf.local_variables_initializer())
		sess.run(init)

		time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
		writer = tf.summary.FileWriter('logs/train:_Block_Sparse_Bayesian')
		test_writer = tf.summary.FileWriter('logs/test:_Block_Sparse_Bayesian')

		general_summaries = tf.summary.merge_all()

		batches = generator([x_train, y_train, np.arange(dataset_size)], 128)
		for i, (batch_x, batch_y, indices) in tqdm(enumerate(batches)):
			feed_dict = {
				inputs: batch_x,
				labels: batch_y,
			}
			_, summary_data = sess.run([opt, general_summaries], feed_dict)
			writer.add_summary(summary_data, global_step=i)

			if i % 100 == 0:
				test_feed_dict = {inputs: x_test, labels: y_test}
				summary_data = sess.run(general_summaries, test_feed_dict)
				test_writer.add_summary(summary_data, global_step=i)

if __name__ == '__main__':
	run()