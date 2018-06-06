import datetime

import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm

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
        # import pdb; pdb.set_trace()
        yield batches


def run():
    """
    Runs the MNIST example
    """
    (x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')
    # x_train = x_train[0:64]
    # y_train = y_train[0:64]

    dataset_size = x_train.shape[0] #Size of the entire training set


    #Placeholders
    inputs = tf.placeholder(tf.float32, [None, 28 * 28], 'inputs')
    labels = tf.placeholder(tf.int32, [None], 'labels')
    data_indices = tf.placeholder(tf.int32, [None], 'data_indices') #Labels the batch...

    def network(context: modular.ModularContext):
        """
        Args:
            Instantiation of the ModularContext class
        """

        modules = modular.create_dense_modules(inputs, module_count=10, units=32, activation=tf.nn.relu) 
        hidden = modular.masked_layer(inputs, modules, context=context) #[sample * B x units]

        modules = modular.create_dense_modules(hidden, module_count=10, units=10, activation=tf.nn.relu)
        logits = modular.masked_layer(hidden, modules, context=context)

        # modules = modular.create_dense_modules(logits, module_count=10, units=10)
        # logits = modular.modular_layer(logits, modules, parallel_count=5, context=context) 

        target = modular.modularize_target(labels, context) #Tile targets 
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target) #Targets are obs, find likelihood

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy

    #make template: create function and partially evaluate it, create variables the first time then
    #reuse them, better than using autoreuse=True in the scope
    template = tf.make_template('network', network)
    optimizer = tf.train.AdamOptimizer()
    e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                              data_indices, sample_size=20)
    ll, logits, accuracy, s_entropy, bs_entropy = eval

    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    with tf.Session() as sess:
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        writer = tf.summary.FileWriter(f'logs/train_masked_trial:1_{time}',sess.graph)
        test_writer = tf.summary.FileWriter(f'logs/test_masked_trial:1_{time}',sess.graph)
        summaries = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        batches = generator([x_train, y_train, np.arange(dataset_size)], 32)
        for i, (batch_x, batch_y, indices) in tqdm(enumerate(batches)):
            feed_dict = {
                inputs: batch_x,
                labels: batch_y,
                data_indices: indices
            }
            step = e_step if i % 15 == 0 else m_step
            _, summary_data = sess.run([step, summaries], feed_dict)
            writer.add_summary(summary_data, global_step=i)

            if i % 100 == 0:
                # print('RUN:',i)
                test_feed_dict = {inputs: x_test, labels: y_test}
                summary_data = sess.run(summaries, test_feed_dict)
                test_writer.add_summary(summary_data, global_step=i)
        writer.close()
        test_writer.close()


if __name__ == '__main__':
    run()
