import datetime
import random
import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm
from tensorflow.python import debug as tf_debug

import numpy as np
from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES, get_tensor_op
from libmodular.layers import create_ema_opt, get_sparsity_level, get_dep_input, get_ctrl_bias, get_ctrl_weights

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

inputs = tf.placeholder(name='x',
                  shape=[batch*2,2],
                  dtype=tf.float32)
labels = tf.placeholder(name='y',
                  shape=[batch*2,2],
                  dtype=tf.int32)

def create_summary(list_of_ops_or_op, name, summary_type):
    summary = getattr(tf.summary, summary_type)

    if type(list_of_ops_or_op) is list:
        for i in range(len(list_of_ops_or_op)):
            summary(str(name) + '_' + str(i), list_of_ops_or_op[i])

    elif type(list_of_ops_or_op) is tf.Tensor:
        summary(str(name), list_of_ops_or_op)

    else:
        raise TypeError('Invalid type for summary')


def create_sparse_summary(sparse_ops):
    def layer_sparsity(op):
        batch_sparse = tf.reduce_sum(op, axis=1)/tf.cast((tf.shape(op)[1]), tf.float32)
        return tf.reduce_mean(batch_sparse)
    sparse_model = tf.reduce_mean([layer_sparsity(op) for op in sparse_ops ])
    create_summary(sparse_model, 'Sparsity ratio', 'scalar')

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
        hidden, l, s, pi, bs = modular.dep_variational_mask(hidden, 
                                                      modules, 
                                                      context, 
                                                      0.001,
                                                      tf.shape(inputs)[0])
        hidden = modular.batch_norm(hidden)
        hidden  = tf.nn.relu(hidden)

        pi_log.append(pi)
        s_log.append(tf.cast(tf.reshape(s, [1,-1,module_count,1]), tf.float32))

        ctrl_logits.append(tf.cast(tf.reshape(l, [1,-1,module_count,1]), tf.float32))
        bs_perst_log.append(tf.cast(tf.reshape(bs, [1,-1,module_count,1]), tf.float32))

    logits = tf.layers.dense(hidden, 2)
    # logits = hidden

    target = modular.modularize_target(labels, context)
    loglikelihood = -tf.losses.mean_squared_error(target, logits)

    loglikelihood = sum_and_mean_il(loglikelihood, context.sample_size)

    # predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))
    accuracy = tf.constant(1)

    return (loglikelihood, ctrl_logits, accuracy,
            bs_perst_log,  s_log, pi_log, context)

template = tf.make_template('network', 
                            network)
optimizer = tf.train.AdamOptimizer(learning_rate=0.05)

iteration_number=tf.placeholder(dtype=tf.float32, shape=[])
num_batches=1.
beta=1
dataset_size=200
data_indices = 1
variational = 'True'
sample_size = 1
epoch_lim = 500.
m_step, eval = modular.modularize_variational(template, optimizer, dataset_size,
                                          data_indices, variational, num_batches, beta,
                                          sample_size, iteration_number, epoch_lim)

ll, ctrl_logits, accuracy,  bs_perst_log, s_log, pi_log, context = eval

params = context.layers
a_list = [l.a for l in params]
b_list = [l.b for l in params]

create_summary(a_list, 'a', 'histogram')
create_summary(b_list, 'b', 'histogram')

create_summary(pi_log, 'pi', 'histogram')
create_summary(ctrl_logits, 'Controller_probs', 'image')
create_summary(s_log, 'Selection', 'image')
create_summary(bs_perst_log, 'Best_selection', 'image')

create_summary(tf.reduce_sum(ll), 'loglikelihood', 'scalar')
create_summary(accuracy, 'accuracy', 'scalar')

create_summary(get_tensor_op(), 'Mod KL', 'scalar')

create_summary(get_dep_input(), 'dep_input', 'histogram')

create_summary(get_ctrl_bias(), 'ctrl_bias', 'histogram')

create_summary(get_ctrl_weights(), 'ctrl_weights', 'histogram')

create_sparse_summary(get_sparsity_level())

with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    step = m_step
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    general_summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(f'toy/TOY_REgression_With_batch_norm_MOD_{time}')

    x_1 = np.random.multivariate_normal(
            mean=np.array([0.0, 0.0]), 
            cov=np.array([[1,0],[0,1]]),
            size=batch)

    x_2 = np.random.multivariate_normal(
            mean=np.array([10.0, 10.0]),
            cov=np.array([[10,0],[0,1]]),
            size=batch)
    full_data = np.concatenate([x_1, x_2])
    target_1 = x_1 @ np.array([[9,0],[0,5]])
    theta=30
    target_2 = rotate((20,20), x_2, math.radians(theta))
    full_target = np.concatenate([target_1, target_2])

    sess.run(init)

    feed_dict = {inputs:full_data,
                labels:full_target}

    j_s=0.    
    for i in tqdm(range(50000)):
        feed_dict[iteration_number] = j_s
        sess.run(step, feed_dict)
        summary = sess.run(general_summaries, feed_dict)
        writer.add_summary(summary, global_step=i)
        if i % 1 == 0 and j_s<epoch_lim-1:
              j_s+=1.
        elif j_s>epoch_lim-1:
            j_s = epoch_lim-1
        else:
              j_s = j_s

writer.close()

