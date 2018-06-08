import datetime

import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

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
        # import 
        yield batches


def run():
    """
    Runs the MNIST example
    """
    times = [0, 2800, 4800, 6000, 8000, 10000, 12000 , 16000, 20000 ]
    plot_logits = []
    (x_train, y_train), (x_test, y_test) = observations.mnist('~/data/MNIST')


    dataset_size = x_train.shape[0] #Size of the entire training set


    #Placeholders
    inputs = tf.placeholder(tf.float32, [None, 28 * 28], 'inputs')
    labels = tf.placeholder(tf.int32, [None], 'labels')
    data_indices = tf.placeholder(tf.int32, [None], 'data_indices') #Labels the batch...

    def network(context: modular.ModularContext, masked_bernoulli=False):
        """
        Args:
            Instantiation of the ModularContext class
        """
        if masked_bernoulli:
            modules = modular.create_dense_modules(inputs, module_count=3, units=128, activation=tf.nn.relu) 
            hidden = modular.masked_layer(inputs, modules, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count=3, units=64, activation=tf.nn.relu) 
            hidden = modular.masked_layer(hidden, modules, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count=3, units=32, activation=tf.nn.relu) 
            hidden = modular.masked_layer(hidden, modules, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count=3, units=10) 
            logits = modular.masked_layer(hidden, modules, context=context) #[sample * B x units]
        else:
            modules = modular.create_dense_modules(inputs, module_count=3, units=128, activation=tf.nn.relu) 
            hidden = modular.modular_layer(inputs, modules, parallel_count=1, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count=3, units=64, activation=tf.nn.relu) 
            hidden = modular.modular_layer(hidden, modules, parallel_count=1, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count=3, units=32, activation=tf.nn.relu) 
            hidden = modular.modular_layer(hidden, modules, parallel_count=1, context=context) #[sample * B x units]

            modules = modular.create_dense_modules(hidden, module_count=3, units=10) 
            logits = modular.modular_layer(hidden, modules, parallel_count=1, context=context) #[sample * B x units]


        target = modular.modularize_target(labels, context) #Tile targets 
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target) #Targets are obs, find likelihood

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy

    #make template: create function and partially evaluate it, create variables the first time then
    #reuse them, better than using autoreuse=True in the scope
    template = tf.make_template('network', network, masked_bernoulli=True)
    optimizer = tf.train.AdamOptimizer()
    e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                              data_indices, sample_size=10)
    ll, logits, accuracy, s_entropy, bs_entropy = eval

    # logits = tf.sigmoid(l)
    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    # try:
    with tf.Session() as sess:
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        writer = tf.summary.FileWriter(f'logs/train_testing:test_no_initial_with_relu_15m_heirarchical_MASKED_random_initialiser_{time}',sess.graph)
        test_writer = tf.summary.FileWriter(f'logs/test_testing:test_no_initial_with_relu_15m_heirarchical_MASKED_random_initialiser_{time}',sess.graph)
        summaries = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        # Initial e-step
        # feed_dict = {
        #         inputs: x_train,
        #         labels: y_train,
        #         data_indices: np.arange(x_train.shape[0])
        #         }
        # sess.run(e_step, feed_dict)


        batches = generator([x_train, y_train, np.arange(dataset_size)], 64)
        for i, (batch_x, batch_y, indices) in tqdm(enumerate(batches)):
            feed_dict = {
                inputs: batch_x,
                labels: batch_y,
                data_indices: indices
            }
            step = e_step if i % 15 == 0 else m_step
            _, summary_data= sess.run([step, summaries], feed_dict)

            # if i in times and 10 in indices:
            #      _, summary_data, logit= sess.run([step, summaries,logits], feed_dict)
            #      plot_logits.append(logit)

            writer.add_summary(summary_data, global_step=i)

            if i % 100 == 0:
                test_feed_dict = {inputs: x_test, labels: y_test}
                summary_data = sess.run(summaries, test_feed_dict)
                test_writer.add_summary(summary_data, global_step=i)
        writer.close()
        test_writer.close()

# except KeyboardInterrupt:
#     print('SHAPE:', len(plot_logits))
#     for i in np.arange(len(plot_logits)):
#         plot = plot_logits[i].T
#         cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['white','black'],256)
#         plt.imshow(plot,interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
#         plt.savefig('plot'+str(i))




if __name__ == '__main__':
    run()
