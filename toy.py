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
import sys
import os

cwd = os.getcwd()

batch=2000

inputs = tf.placeholder(name='x',
                  shape=[batch*2,2],
                  dtype=tf.float32)
labels = tf.placeholder(name='y',
                  shape=[batch*2],
                  dtype=tf.int32)

# b = tf.get_variable(name='bias_linear',
#                    shape=[2],
#                    dtype=tf.float32)
# W = tf.get_variable(name='weight_linear',
#                    shape=[2,2],
#                    dtype=tf.float32,
#                    initializer=tf.contrib.layers.xavier_initializer())

# hidden_true = tf.matmul(inputs,W) + b

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
    units = [100]
    layers = len(units)
    s_log = []
    ctrl_logits =[]
    pi_log = []
    bs_perst_log = []
    module_count = 3



    for i in range(layers):

      modules = modular.create_dense_modules(hidden, 
                                              module_count, 
                                              units=units[i], 
                                              activation=tf.nn.relu) 
      hidden, l, s, pi, bs = modular.variational_mask(hidden, 
                                                      modules, 
                                                      context, 
                                                      0.001,
                                                      tf.shape(inputs)[0])
      pi_log.append(pi)
      s_log.append(tf.cast(tf.reshape(s, [1,-1,module_count,1]), tf.float32))

    ctrl_logits.append(tf.cast(tf.reshape(l, [1,-1,module_count,1]), tf.float32))
    bs_perst_log.append(tf.cast(tf.reshape(bs, [1,-1,module_count,1]), tf.float32))

    logits = tf.layers.dense(hidden, 10)

    target = modular.modularize_target(labels, context)
    loglikelihood = tf.distributions.Categorical(logits).log_prob(target)

    loglikelihood = sum_and_mean_il(loglikelihood, context.sample_size)

    predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

    return (loglikelihood, ctrl_logits, accuracy,
            bs_perst_log,  s_log, pi_log, context)

template = tf.make_template('network', 
                            network)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

iteration_number=tf.placeholder(dtype=tf.float32, shape=[])
num_batches=1.
beta=1
dataset_size=200
data_indices = 1
variational = 'True'
sample_size = 5
epoch_lim = 600.
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

# saver = tf.train.Saver({'bias_linear':b,
#                         'weight_linear':W})

with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    step = m_step
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    general_summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(f'toy/TOY_Independent_Separable_2unitmodules_FINAL_{time}')
    data_1 = np.random.normal(loc=0.0, scale=1.0, size=(batch, 2))
    data_2 = np.random.normal(loc=10.0, scale=1.0, size=(batch, 2))
    full_data = np.concatenate([data_1, data_2], axis=0)
    label_1 = np.zeros(batch, dtype=int)
    label_2 = np.ones(batch, dtype=int) 
    full_label = np.concatenate([label_1, label_2], axis=0)
    sess.run(init)
    # saver.restore(sess, cwd+"/toy_weights/model.ckpt")


    feed_dict = {inputs:full_data,
                labels:full_label}

    j_s=0.    
    for i in tqdm(range(40000)):
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

