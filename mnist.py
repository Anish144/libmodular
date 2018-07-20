import datetime
import random
import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pickle
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

def run():
    """
    Runs the MNIST example
    """
    (x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')


    dataset_size = x_train.shape[0] #Size of the entire training set


    #Placeholders
    inputs = tf.placeholder(tf.float32, [None, 28 * 28], 'inputs')
    labels = tf.placeholder(tf.int32, [None], 'labels')
    data_indices = tf.placeholder(tf.int32, [None], 'data_indices') #Labels the batch...

    module_count = 10
    variational = True
    masked_bernoulli = False
    new_controller = False

    def network(context: modular.ModularContext, masked_bernoulli=False, variational=variational):
        """
        Args:
            Instantiation of the ModularContext class
        """
        if masked_bernoulli:

            modules = modular.create_dense_modules(inputs, module_count, units=128, activation=tf.nn.relu) 
            hidden, l1, bs_1 = modular.masked_layer(inputs, modules, context,  get_initialiser(dataset_size, 2, module_count)) #[sample * B x units]

            # modules = modular.create_dense_modules(hidden, module_count, units=64, activation=tf.nn.relu) 
            # hidden, l2, _ = modular.masked_layer(hidden, modules, context,  get_initialiser(dataset_size, 2, module_count)) #[sample * B x units]

            # modules = modular.create_dense_modules(hidden, module_count, units=32, activation=tf.nn.relu) 
            # hidden, l3, _ = modular.masked_layer(hidden, modules, context,  get_initialiser(dataset_size, 2, module_count)) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=10) 
            logits, l2, bs = modular.masked_layer(hidden, modules, context,  get_initialiser(dataset_size, 2, module_count)) #[sample * B x units]

        elif variational:

            modules = modular.create_dense_modules(inputs, module_count, units=128, activation=tf.nn.relu) 
            hidden, l1, bs_1 = modular.variational_mask(inputs, modules, context, 0.001, 3.17) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=64, activation=tf.nn.relu) 
            hidden, l2, bs_2 = modular.variational_mask(hidden, modules, context, 0.001, 3.17) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=32, activation=tf.nn.relu) 
            hidden, l3, bs_3 = modular.variational_mask(hidden, modules, context, 0.001, 3.17) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=10) 
            logits, l4, bs_4 = modular.variational_mask(hidden, modules, context, 0.001, 3.17) #[sample * B x units]

        elif new_controller:

            modules = modular.create_dense_modules(inputs, module_count, units=128, activation=tf.nn.relu) 
            hidden, l1, bs_1 = modular.new_controller(inputs, modules, context,  get_initialiser(dataset_size, 5, module_count)) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=64, activation=tf.nn.relu) 
            hidden, l2, bs_2 = modular.new_controller(hidden, modules, context,  get_initialiser(dataset_size, 5, module_count)) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=32, activation=tf.nn.relu) 
            hidden, l3, bs_3 = modular.new_controller(hidden, modules, context,  get_initialiser(dataset_size, 5, module_count)) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=10) 
            logits, l4, bs_4 = modular.new_controller(hidden, modules, context,  get_initialiser(dataset_size, 5, module_count)) #[sample * B x units]

        else:

            modules = modular.create_dense_modules(inputs, module_count, units=128, activation=tf.nn.relu) 
            hidden, l1, _= modular.modular_layer(inputs, modules, parallel_count=3, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=64, activation=tf.nn.relu) 
            hidden, l2, _ = modular.modular_layer(hidden, modules, parallel_count=3, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=32, activation=tf.nn.relu) 
            hidden, l3, _ = modular.modular_layer(hidden, modules, parallel_count=3, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=10) 
            logits, l4, bs = modular.modular_layer(hidden, modules, parallel_count=3, context=context) #[sample * B x units]


        target = modular.modularize_target(labels, context) #Tile targets 
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target) #Targets are obs, find likelihood

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy, bs_1, bs_2, bs_3, bs_4, l1, l2, l3, l4

    #make template: create function and partially evaluate it, create variables the first time then
    #reuse them, better than using autoreuse=True in the scope
    template = tf.make_template('network', network, masked_bernoulli=masked_bernoulli, variational=variational)
    optimizer = tf.train.AdamOptimizer()

    if variational:
        m_step, eval = modular.modularize_variational(template, optimizer, dataset_size,
                                                  data_indices, variational=variational)
    else:
        e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                                  data_indices, sample_size=10, variational=variational)
    ll, logits, accuracy, s_entropy, bs_entropy, bs_1, bs_2, bs_3, bs_4, l1, l2, l3, l4 = eval

    # bs_1 = tf.reshape(bs_1, [1,-1,module_count,1])
    # bs_2 = tf.reshape(bs_2, [1,-1,module_count,1])
    # bs_3 = tf.reshape(bs_3, [1,-1,module_count,1])
    # bs_4 = tf.reshape(bs_4, [1,-1,module_count,1])

    l1_re = tf.reshape(tf.cast(l1, tf.float32), [1,-1,module_count,1])
    l2_re = tf.reshape(tf.cast(l2, tf.float32), [1,-1,module_count,1])
    l3_re = tf.reshape(tf.cast(l3, tf.float32), [1,-1,module_count,1])
    l4_re = tf.reshape(tf.cast(l4, tf.float32), [1,-1,module_count,1])

    # tf.summary.image('best_selection_1', tf.cast(bs_1, dtype=tf.float32), max_outputs=1)
    # tf.summary.image('best_selection_2', tf.cast(bs_2, dtype=tf.float32), max_outputs=1)
    # tf.summary.image('best_selection_3', tf.cast(bs_3, dtype=tf.float32), max_outputs=1)
    # tf.summary.image('best_selection_4', tf.cast(bs_4, dtype=tf.float32), max_outputs=1)

    tf.summary.image('l1_controller_probs', l1_re, max_outputs=1)
    tf.summary.image('l2_controller_probs', l2_re, max_outputs=1)
    tf.summary.image('l3_controller_probs', l3_re, max_outputs=1)
    tf.summary.image('l4_controller_probs', l4_re, max_outputs=1)

    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    try:
        with tf.Session() as sess:
            time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())

            if REALRUN=='True':
                writer = tf.summary.FileWriter(f'logs/train:_10m_Variational_try_Initial:20_alpha:1_{time}',sess.graph)
                test_writer = tf.summary.FileWriter(f'logs/test:_10m_Variational_try_Initial:20_alpha:1_{time}',sess.graph)

            general_summaries = tf.summary.merge_all()
            m_step_summaries = tf.summary.merge([create_m_step_summaries(), general_summaries])
            sess.run(tf.global_variables_initializer())

            # Initial e-step
            if not variational:
                feed_dict = {
                        inputs: x_train,
                        labels: y_train,
                        data_indices: np.arange(x_train.shape[0])
                        }
                sess.run(e_step, feed_dict)

            batches = generator([x_train, y_train, np.arange(dataset_size)], 300)
            for i, (batch_x, batch_y, indices) in tqdm(enumerate(batches)):
                feed_dict = {
                    inputs: batch_x,
                    labels: batch_y,
                    data_indices: indices,
                }
                if variational:
                    step = m_step
                else:
                    step = e_step if i % 10 == 0 else m_step
                _, summary_data, log = sess.run([step, m_step_summaries, logits], feed_dict)

                if REALRUN=='True':
                    writer.add_summary(summary_data, global_step=i)

                if i % 100 == 0:
                    test_feed_dict = {inputs: x_test, labels: y_test, data_indices: np.arange(x_test.shape[0])}
                    summary_data, ctrl = sess.run([m_step_summaries, l1], test_feed_dict)

                    if REALRUN=='True':
                        test_writer.add_summary(summary_data, global_step=i)

            writer.close()
            test_writer.close()

    except KeyboardInterrupt:
        pass




if __name__ == '__main__':
    run()
