import datetime

import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm
import sys
from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES


def fix_image_summary(list_op, op, module_count):
    list_op.append(
        tf.cast(
            tf.reshape(
                op,
                [1, -1, module_count, 1]),
            tf.float32))


pass


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
    batch = 25
    train = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))._enumerate().repeat().shuffle(50000).batch(batch)
    # Test dataset
    test_batch_size = 25
    # dummy_data_indices = tf.zeros([test_batch_size], dtype=tf.int64)
    test = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test))._enumerate().repeat().batch(test_batch_size)

    # Handle to switch between datasets
    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(
        handle, train.output_types, train.output_shapes)
    data_indices, (inputs, labels) = itr.get_next()

    # Preprocessing
    inputs_cast = tf.cast(inputs, tf.float32) / 255.0
    inputs_tr = tf.transpose(inputs_cast, perm=(0, 2, 3, 1))
    labels_cast = tf.cast(labels, tf.int32)

    CNN_module_number = [4, 4, 4, 4]
    filter_list = [16, 16, 32, 32]
    linear_module_number = [8, 4]

    def network(context: modular.ModularContext, masked_bernoulli=False):
        # 4 modular CNN layers
        activation = inputs_tr
        logit = []
        bs_list = []
        for j in range(len(CNN_module_number)):
            input_channels = activation.shape[-1]
            out_filter = filter_list[j]
            filter_shape = [5, 5, input_channels, out_filter]
            module_count = CNN_module_number[j]
            modules = modular.create_conv_modules(
                filter_shape, module_count, strides=[1, 1, 1, 1])
            if not masked_bernoulli:
                hidden, l, bs = modular.modular_layer(
                    activation, modules, parallel_count=parallel[j], context=context)
                l = tf.reshape(
                    tf.cast(
                        tf.nn.softmax(l), tf.float32), [1, -1, module_count, 1])
                logit.append(l)
                bs = tf.reshape(tf.cast(bs, tf.float32),
                                [1, -1, module_count, 1])
                bs_list.append(bs)
            else:
                hidden, l, bs = modular.masked_layer(
                    activation,
                    modules,
                    context,
                    get_initialiser([dataset_size, module_count], 0.5))
                fix_image_summary(logit, l, module_count)
                fix_image_summary(bs_list, bs, module_count)
            hidden = tf.nn.max_pool(
                hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation = tf.nn.relu(hidden)
            activation = modular.batch_norm(activation)

        flattened = tf.layers.flatten(activation)

        for j in range(len(linear_module_number)):
            module_count = linear_module_number[j]
            modules = modular.create_dense_modules(
                flattened, module_count, units=48)
            hidden, l, bs = modular.masked_layer(
                flattened,
                modules,
                context,
                get_initialiser([dataset_size, module_count], 0.75))
            fix_image_summary(logit, l, module_count)
            fix_image_summary(bs_list, bs, module_count)
            flattened = modular.batch_norm(flattened)

        logits = tf.layers.dense(flattened, units=10)

        target = modular.modularize_target(labels_cast, context)
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target)

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy, logit, bs_list

    template = tf.make_template('network', network, masked_bernoulli=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                              data_indices, sample_size=10)
    ll, logits, accuracy, s_entropy, bs_entropy, logit, bs_list = eval

    create_summary(logit, 'Controller Probs', 'image')

    create_summary(bs_list, 'Best Selection', 'image')

    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))
    create_summary(logit, 'Controller', 'histogram')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        writer = tf.summary.FileWriter(
            (f'logs/train_Tutorial_Bernoulli_mask_EM_Func_add_1step_:' +
             f'_{time}'),
            sess.graph)
        test_writer = tf.summary.FileWriter(
            (f'logs/test_Tutorial_Bernoulli_mask_EM_Func_add_1step_:' +
             f'_{time}'),
            sess.graph)
        general_summaries = tf.summary.merge_all()
        m_step_summaries = tf.summary.merge(
            [create_m_step_summaries(), general_summaries])
        sess.run(tf.global_variables_initializer())
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}

        # Initial e-step
        for _ in range(dataset_size // batch):
            sess.run(e_step, train_dict)

        for i in tqdm(range(100000)):
            # Switch between E-step and M-step
            step = e_step if i % 1 == 0 else m_step

            # Sometimes generate summaries
            if i % 52 == 0:
                summaries = m_step_summaries if step == m_step else general_summaries
                _, summary_data = sess.run([step, summaries], train_dict)
                writer.add_summary(summary_data, global_step=i)
                summary_data = sess.run(general_summaries, test_dict)
                test_writer.add_summary(summary_data, global_step=i)

                accuracy_log = []
                for test in range(x_test.shape[0] // test_batch_size):
                    test_accuracy = sess.run(accuracy, test_dict)
                    accuracy_log.append(test_accuracy)
                final_accuracy = np.mean(accuracy_log)
                summary = tf.Summary()
                summary.value.add(tag='Test Accuracy',
                                  simple_value=final_accuracy)

                test_writer.add_summary(summary, global_step=i)
            else:
                sess.run(step, train_dict)

        writer.close()
        test_writer.close()


if __name__ == '__main__':
    run()
