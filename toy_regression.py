import datetime
import random
import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm
from tensorflow.python import debug as tf_debug

import numpy as np
from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES
import sys

import math
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px = point[:,0]
    py = point[:,1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    final = np.zeros((point.shape[0], point.shape[1]))
    final[:,0] = qx
    final[:,1] = qy
    return final



batch=1000

x_1 = np.random.multivariate_normal(
        mean=np.array([-5.0, -5.0]), 
        cov=np.array([[1,0],[0,1]]),
        size=batch)

x_2 = np.random.multivariate_normal(
        mean=np.array([20.0, 20.0]),
        cov=np.array([[10,0],[0,1]]),
        size=batch)
full_data = np.concatenate([x_1, x_2])
target_1 = x_1 @ np.array([[9,0],[0,5]])
theta=30
target_2 = rotate((20,20), x_2, math.radians(theta))
full_target = np.concatenate([target_1, target_2])

dataset_size = batch*2 #Size of the entire training set


inputs = tf.placeholder(name='x',
                  shape=[batch*2,2],
                  dtype=tf.float32)
labels = tf.placeholder(name='y',
                  shape=[batch*2,2],
                  dtype=tf.int32)
data_indices = tf.placeholder(tf.int32, [None], 'data_indices') #Labels the batch...


def create_summary(list_of_ops_or_op, name, summary_type):
    summary = getattr(tf.summary, summary_type)

    if type(list_of_ops_or_op) is list:
        for i in range(len(list_of_ops_or_op)):
            summary(str(name) + '_' + str(i), list_of_ops_or_op[i])

    elif type(list_of_ops_or_op) is tf.Tensor:
        summary(str(name), list_of_ops_or_op)

    else:
        raise TypeError('Invalid type for summary')

def sum_and_mean_il(il, sample_size):
    il = tf.reshape(il, [-1,
                        sample_size])
    il = tf.reduce_sum(il, axis=0)
    return tf.reduce_mean(il, axis=0)

def network(context: modular.ModularContext):
    """
    Args:
        Instantiation of the ModularContext class
    """
    hidden = inputs
    units = [2]
    layers = len(units)
    s_log = []
    ctrl_logits =[]
    pi_log = []
    bs_perst_log = []
    module_count = 2

    for i in range(layers):

      modules = modular.create_dense_modules(hidden, 
                                              module_count, 
                                              units=units[i], 
                                              activation=tf.nn.relu) 
      hidden, l, bs = modular.modular_layer(hidden, 
                                          modules,
                                          1,
                                          context)

    ctrl_logits.append(tf.cast(tf.reshape(l, [1,-1,module_count,1]), tf.float32))
    bs_perst_log.append(tf.cast(tf.reshape(bs, [1,-1,module_count,1]), tf.float32))

    logits = hidden

    target = modular.modularize_target(labels, context)
    loglikelihood = - tf.einsum('bi,bi->b', tf.cast(target, tf.float32) - logits, tf.cast(target, tf.float32) - logits)
    # import pdb; pdb.set_trace()

    # predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))
    accuracy = tf.constant(1)

    selection_entropy = context.selection_entropy()
    batch_selection_entropy = context.batch_selection_entropy()

    return (loglikelihood, ctrl_logits, accuracy,
            bs_perst_log,  s_log, pi_log, context, selection_entropy,
            batch_selection_entropy)

template = tf.make_template('network', 
                            network)
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)

iteration_number=tf.placeholder(dtype=tf.float32, shape=[])

e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                          data_indices, sample_size=10)

ll, ctrl_logits, accuracy,  bs_perst_log, s_log, pi_log, context, selection_entropy, batch_selection_entropy = eval

params = context.layers

create_summary(selection_entropy, 'selection entropy', 'scalar')
create_summary(batch_selection_entropy, 'batch selection entropy', 'scalar')

create_summary(ctrl_logits, 'Controller_probs', 'image')
create_summary(bs_perst_log, 'Best_selection', 'image')

create_summary(tf.reduce_mean(ll), 'loglikelihood', 'scalar')


with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    step = m_step
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    general_summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(f'toy/TOY_REgression_EMVIT_{time}')

    sess.run(init)

    feed_dict = {inputs:full_data,
                labels:full_target,
                data_indices:np.arange(dataset_size)}

    for i in tqdm(range(50000)):
        sess.run(step, feed_dict)
        summary = sess.run(general_summaries, feed_dict)
        writer.add_summary(summary, global_step=i)

writer.close()

