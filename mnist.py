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

REALRUN = sys.argv[1]

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

def get_initialiser(data_size, n, module_count):
    choice = np.zeros((data_size, n), dtype=int)
    for j in range(data_size):
        choice[j,:] = random.sample(range(module_count), n)
    one_hot = np.zeros((data_size, module_count), dtype=int)
    for i in range(n):
        one_hot[np.arange(data_size), choice[:,i]]=1
    return tf.constant_initializer(one_hot, dtype=tf.int32, verify_shape=True)

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

def run():
    """
    Runs the MNIST example
    """
    (x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')


    dataset_size = x_train.shape[0] #Size of the entire training set

    batch_size = 32
    num_batches = dataset_size/batch_size

    #Placeholders
    inputs = tf.placeholder(tf.float32, [None, 28 * 28], 'inputs')
    labels = tf.placeholder(tf.int32, [None], 'labels')
    data_indices = tf.placeholder(tf.int32, [None], 'data_indices') #Labels the batch...

    module_count = 16
    variational = 'True'
    masked_bernoulli = False
    new_controller = False
    epoch_lim = 8

    iteration = tf.placeholder(name='iteration',
                        shape=[],
                        dtype=tf.float32)

    beta = 1.
    sample_size = 2

    iteration_number = tf.placeholder(dtype=tf.float32,
                                shape=[],
                                name='iteration_number')

    def network(context: modular.ModularContext, variational=variational):
        """
        Args:
            Instantiation of the ModularContext class
        """
        hidden = inputs
        module_count_list = [80, 40, 20, 5]
        layers = len(module_count_list)
        s_log = []
        ctrl_logits =[]
        pi_log = []
        bs_perst_log = []

        for i in range(layers):
            module_count = module_count_list[i]
            if masked_bernoulli:

                modules = modular.create_dense_modules(hidden, 
                                                        module_count, 
                                                        units=units[i],
                                                        activation=tf.nn.relu) 
                hidden, l, bs = modular.masked_layer(hidden,
                                                        modules,
                                                        context,
                                                        get_initialiser(dataset_size, 2, module_count))

            elif variational == 'True':

                modules = modular.create_dense_modules(hidden, 
                                                        module_count, 
                                                        units=16, 
                                                        activation=tf.nn.relu) 
                hidden, l, s, bs, pi = modular.dep_variational_mask(hidden, 
                                                                modules, 
                                                                context, 
                                                                0.001,
                                                                tf.shape(inputs)[0])
                pi_log.append(pi)
                s_log.append(tf.cast(tf.reshape(s, [1,-1,module_count,1]), tf.float32))

            else:

                modules = modular.create_dense_modules(hidden, 
                                                        module_count, 
                                                        units=units[i],
                                                        activation=tf.nn.relu) 
                hidden, l, bs = modular.modular_layer(hidden,
                                                        modules, 
                                                        parallel_count=1, 
                                                        context=context)

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
                                network, 
                                variational=variational)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)

    if variational == 'True':
        m_step, eval = modular.modularize_variational(template, optimizer, dataset_size,
                                                  data_indices, variational, num_batches, beta,
                                                  sample_size, iteration_number, epoch_lim)
    else:
        e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                                  data_indices, sample_size=10, variational=variational)
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

    try:
        with tf.Session() as sess:
            time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            if REALRUN=='True':
                writer = tf.summary.FileWriter(f'logs/train:_4layer_MNIST_a:.1.5_b:0.1_alpha:0.1_beta:1_lr:0.0005_DEPENDENT_MASK_{time}',
                                                sess.graph)
                test_writer = tf.summary.FileWriter(f'logs/test:_4layer_MNIST_a:1.5_b:0.1_alpha:0.1_beta:1_lr:0.0005_DEPENDENT_MASK_{time}',
                                                    sess.graph)
            general_summaries = tf.summary.merge_all()
            m_step_summaries = tf.summary.merge([create_m_step_summaries(), general_summaries])
            sess.run(tf.global_variables_initializer())

            # Initial e-step
            if variational == 'False':
                feed_dict = {
                        inputs: x_train,
                        labels: y_train,
                        data_indices: np.arange(x_train.shape[0])
                        }
                sess.run(e_step, feed_dict)

            j_s=0.    
            batches = generator([x_train, y_train, np.arange(dataset_size)], batch_size)
            for i, (batch_x, batch_y, indices) in tqdm(enumerate(batches)):
                feed_dict = {
                    inputs: batch_x,
                    labels: batch_y,
                    data_indices: indices,
                    iteration_number: j_s
                }
                if variational == 'True':
                    step = m_step
                else:
                    step = e_step if i % 10 == 0 else m_step
                _, summary_data = sess.run([step, m_step_summaries], feed_dict)

                if REALRUN=='True':
                    writer.add_summary(summary_data, global_step=i)

                # if i % 400 == 0:
                #     test_feed_dict = {inputs: x_test, labels: y_test, data_indices: np.arange(x_test.shape[0]),
                #                         iteration_number: j}
                                        
                #     summary_data = sess.run(m_step_summaries, test_feed_dict)

                #     if REALRUN=='True':
                #         test_writer.add_summary(summary_data, global_step=i)
                if i % (dataset_size//batch_size) == 0 and j_s<epoch_lim-1:
                    j_s+=1.
                elif j_s>epoch_lim-1:
                    j_s = epoch_lim-1
                else:
                    j_s = j_s
            writer.close()
            test_writer.close()

    except KeyboardInterrupt:
        pass




if __name__ == '__main__':
    run()
