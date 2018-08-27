import datetime

import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm

from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES


def create_summary(list_of_ops_or_op, name, summary_type):
    summary = getattr(tf.summary, summary_type)

    if type(list_of_ops_or_op) is list:
        for i in range(len(list_of_ops_or_op)):
            summary(str(name) + '_' + str(i), list_of_ops_or_op[i])

    elif type(list_of_ops_or_op) is tf.Tensor:
        summary(str(name), list_of_ops_or_op)

    else:
        raise TypeError('Invalid type for summary')

def get_initialiser(shape, p):
    init = np.random.binomial(n=1, p=p, size=shape)
    return tf.constant_initializer(init, dtype=tf.int32, verify_shape=True)

def make_handle(sess, dataset):
    iterator = dataset.make_initializable_iterator()
    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle


# noinspection PyProtectedMember
def run():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = observations.cifar10('~/data/cifar10')
    y_test = y_test.astype(np.uint8)  # Fix test_data dtype
    dataset_size = x_train.shape[0]

    # Train dataset
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))._enumerate().repeat().shuffle(50000).batch(128)
    # Test dataset
    test_batch_size = 500
    # dummy_data_indices = tf.zeros([test_batch_size], dtype=tf.int64)
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))._enumerate().repeat().batch(test_batch_size)

    # Handle to switch between datasets
    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(handle, train.output_types, train.output_shapes)
    data_indices, (inputs, labels) = itr.get_next()

    # Preprocessing
    inputs_cast = tf.cast(inputs, tf.float32) / 255.0
    inputs_tr = tf.transpose(inputs_cast, perm=(0, 2, 3, 1))
    labels_cast = tf.cast(labels, tf.int32)

    module_count = 5

    def network(context: modular.ModularContext, masked_bernoulli=False):
        # 4 modular CNN layers
        activation = inputs_tr
        logit=[]
        bs_list = []
        for j in range(2):
            input_channels = activation.shape[-1]
            filter_shape = [3, 3, input_channels, 8]
            modules = modular.create_conv_modules(filter_shape, module_count, strides=[1, 2, 2, 1])
            if not masked_bernoulli:
                hidden, l, bs = modular.modular_layer(activation, modules, parallel_count=3, context=context)
                l = tf.reshape(tf.cast(tf.nn.softmax(l), tf.float32), [1,-1,module_count,1])
                logit.append(l)
                bs = tf.reshape(tf.cast(bs, tf.float32), [1,-1,module_count,1])
                bs_list.append(bs)
            else:
                hidden, l, bs = modular.masked_layer(activation, modules, context, get_initialiser([dataset_size, module_count], 0.25))
                logit.append(tf.sigmoid(l))
            pooled = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation = tf.nn.relu(hidden)

        flattened = tf.layers.flatten(activation)
        logits = tf.layers.dense(flattened, units=10)

        target = modular.modularize_target(labels_cast, context)
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target)

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy, logit, bs_list

    template = tf.make_template('network', network, masked_bernoulli=False)
    optimizer = tf.train.AdamOptimizer()
    e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                              data_indices, sample_size=10)
    ll, logits, accuracy, s_entropy, bs_entropy, logit, bs_list = eval

    create_summary(logit, 'Controller Probs', 'image')

    create_summary(bs_list, 'Best Selection', 'image')

    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        writer = tf.summary.FileWriter(f'logs/train_5m_withsummary_EM_VIT_2layer_3parallel_{time}', sess.graph)
        test_writer = tf.summary.FileWriter(f'logs/test_5m_withsummary_EM_VIT_2layer_3parallel_{time}', sess.graph)
        general_summaries = tf.summary.merge_all()
        m_step_summaries = tf.summary.merge([create_m_step_summaries(), general_summaries])
        sess.run(tf.global_variables_initializer())
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}

        # Initial e-step
        # feed_dict = {
        #         inputs: x_train,
        #         labels: y_train,
        #         data_indices: np.arange(x_train.shape[0])
        #         }
        # sess.run(e_step, feed_dict)

        for i in tqdm(range(500000)):
            # Switch between E-step and M-step
            step = e_step if i % 50 == 0 else m_step

            # Sometimes generate summaries
            if i % 50 == 0:
                summaries = m_step_summaries if step == m_step else general_summaries
                _, summary_data = sess.run([step, summaries], train_dict)
                writer.add_summary(summary_data, global_step=i) 
                summary_data = sess.run(general_summaries, test_dict)
                test_writer.add_summary(summary_data, global_step=i)

                accuracy_log = []
                for test in range(x_test.shape[0]//test_batch_size):
                    test_accuracy = sess.run(accuracy, test_dict)
                    accuracy_log.append(test_accuracy)
                final_accuracy = np.mean(accuracy_log)
                summary = tf.Summary()
                summary.value.add(tag='Test Accuracy', 
                                  simple_value = final_accuracy)

                test_writer.add_summary(summary, global_step=i)
            else:
                sess.run(step, train_dict)


        writer.close()
        test_writer.close()


if __name__ == '__main__':
    run()
