import datetime

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

def get_initialiser(shape, p):
    init = np.random.binomial(n=1, p=p, size=shape)
    return tf.constant_initializer(init, dtype=tf.int32, verify_shape=True)

def run():
    """
    Runs the MNIST example
    """
    plot_logits = []
    (x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')


    dataset_size = x_train.shape[0] #Size of the entire training set


    #Placeholders
    inputs = tf.placeholder(tf.float32, [None, 28 * 28], 'inputs')
    labels = tf.placeholder(tf.int32, [None], 'labels')
    data_indices = tf.placeholder(tf.int32, [None], 'data_indices') #Labels the batch...

    module_count = 5

    def network(context: modular.ModularContext, masked_bernoulli=False):
        """
        Args:
            Instantiation of the ModularContext class
        """
        if masked_bernoulli:
            # import pdb; pdb.set_trace()
            modules = modular.create_dense_modules(inputs, module_count, units=128, activation=tf.nn.relu) 
            hidden, l1, _ = modular.masked_layer(inputs, modules, context, get_initialiser([dataset_size, module_count], 0.65)) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=64, activation=tf.nn.relu) 
            hidden, l2, _ = modular.masked_layer(hidden, modules, context, get_initialiser([dataset_size, module_count], 0.65)) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=32, activation=tf.nn.relu) 
            hidden, l3, _ = modular.masked_layer(hidden, modules, context, get_initialiser([dataset_size, module_count], 0.65)) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=10) 
            logits, l4, bs = modular.masked_layer(hidden, modules, context, get_initialiser([dataset_size, module_count], 0.65)) #[sample * B x units]
            # import pdb; pdb.set_trace()

        else:
            modules = modular.create_dense_modules(inputs, module_count, units=128, activation=tf.nn.relu) 
            hidden, l1= modular.modular_layer(inputs, modules, parallel_count=3, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=64, activation=tf.nn.relu) 
            hidden, l2 = modular.modular_layer(hidden, modules, parallel_count=3, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=32, activation=tf.nn.relu) 
            hidden, l3 = modular.modular_layer(hidden, modules, parallel_count=3, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count, units=10) 
            logits, l4 = modular.modular_layer(hidden, modules, parallel_count=3, context=context) #[sample * B x units]


        target = modular.modularize_target(labels, context) #Tile targets 
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target) #Targets are obs, find likelihood

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy ,tf.sigmoid(l4), tf.sigmoid(l3), tf.sigmoid(l2), tf.sigmoid(l1), bs

    #make template: create function and partially evaluate it, create variables the first time then
    #reuse them, better than using autoreuse=True in the scope
    # shape = [dataset_size, module_count]
    # initializer = tf.placeholder(tf.int32, shape, 'init_for_mask')
    template = tf.make_template('network', network, masked_bernoulli=True)
    optimizer = tf.train.AdamOptimizer()
    e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                              data_indices, sample_size=10)
    ll, logits, accuracy, s_entropy, bs_entropy, l4, l3, l2, l1, bs = eval

    bs = tf.reshape(bs, [1,-1,module_count,1])
    l1 = tf.reshape(l1, [1,-1,module_count,1])
    l2 = tf.reshape(l2, [1,-1,module_count,1])
    l3 = tf.reshape(l3, [1,-1,module_count,1])
    l4 = tf.reshape(l4, [1,-1,module_count,1])
    tf.summary.image('best_selection', tf.cast(bs, dtype=tf.float32), max_outputs=1)
    tf.summary.image('l1_controller_probs', l1, max_outputs=1)
    tf.summary.image('l2_controller_probs', l2, max_outputs=1)
    tf.summary.image('l3_controller_probs', l3, max_outputs=1)
    tf.summary.image('l4_controller_probs', l4, max_outputs=1)
    # tf.summary.image('best_selection_persistent', bsp, max_outputs=1)
    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    try:
        with tf.Session() as sess:
            time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
            writer = tf.summary.FileWriter(f'logs/train_testing:Masked_25m_0.65initialiser_initialE_{time}',sess.graph)
            test_writer = tf.summary.FileWriter(f'logs/test_testing:Masked_25m_0.65initialiser_initialE_{time}',sess.graph)
            summaries = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            # Initial e-step
            feed_dict = {
                    inputs: x_train,
                    labels: y_train,
                    data_indices: np.arange(x_train.shape[0])
                    }
            sess.run(e_step, feed_dict)

            batches = generator([x_train, y_train, np.arange(dataset_size)], 128)
            for i, (batch_x, batch_y, indices) in tqdm(enumerate(batches)):
                feed_dict = {
                    inputs: batch_x,
                    labels: batch_y,
                    data_indices: indices,
                }
                step = e_step if i % 25 == 0 else m_step
                _, summary_data, log= sess.run([step, summaries, logits], feed_dict)



                writer.add_summary(summary_data, global_step=i)

                if i % 100 == 0:
                    test_feed_dict = {inputs: x_test, labels: y_test, data_indices: np.arange(x_test.shape[0])}
                    summary_data = sess.run(summaries, test_feed_dict)
                    test_writer.add_summary(summary_data, global_step=i)
            writer.close()
            test_writer.close()

    except KeyboardInterrupt:
        pass




if __name__ == '__main__':
    run()
