import datetime
import random
import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm
import sys
from tensorflow.python import debug as tf_debug
# import matplotlib.pyplot as plt
from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES, get_tensor_op, get_op, get_KL
from libmodular.layers import create_ema_opt

REALRUN = sys.argv[1]


def make_handle(sess, dataset):
    iterator = dataset.make_initializable_iterator()
    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle


def get_dataset(x, y, batch_size):
    data =  tf.data.Dataset.from_tensor_slices((x, y))
    prepare = data._enumerate().repeat().shuffle(50000)
    return prepare.batch(batch_size)


def create_summary(list_of_ops_or_op, name, summary_type):
    summary = getattr(tf.summary, summary_type)

    if type(list_of_ops_or_op) is list:
        for i in range(len(list_of_ops_or_op)):
            summary(str(name) + '_' + str(i), list_of_ops_or_op[i])

    elif type(list_of_ops_or_op) is tf.Tensor:
        summary(str(name), list_of_ops_or_op)

    else:
        raise TypeError('Invalid type for summary')


# noinspection PyProtectedMember
def run():
     # Load dataset
    (x_train_1, y_train), (x_test_1, y_test_1) = observations.cifar10(
        '~/data/cifar10')
    y_test = y_test_1.astype(np.uint8)  # Fix test_data dtype

    # (x_train_2, y_train_2), (x_test_2, y_test_2) = observations.svhn(
    #     '~/data/svhn')
    # y_test_2 = y_test_2.astype(np.uint8)  # Fix test_data dtype

    # Preprocessing
    x_train = np.transpose(x_train_1, [0, 2, 3, 1])
    x_test = np.transpose(x_test_1, [0, 2, 3, 1])

    # x_train = np.concatenate([x_train_1, x_train_2])
    # y_train = np.concatenate([y_train_1, y_train_2])
    # x_test = np.concatenate([x_test_1, x_test_2])
    # y_test = np.concatenate([y_test_1, y_test_2])

    dataset_size = x_train.shape[0]

    batch_size = 50
    num_batches = dataset_size / batch_size

    # Train dataset
    train = get_dataset(x_train, y_train, batch_size)

    # Test dataset
    test_batch_size = 100
    test = get_dataset(x_test, y_test, test_batch_size)

    # Handle to switch between datasets
    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(
        handle, train.output_types, train.output_shapes)
    data_indices, (inputs, labels) = itr.get_next()

    inputs_tr = tf.cast(inputs, tf.float32) / 255.0
    labels_cast = tf.cast(labels, tf.int32)

    def network():
        # 4 modular CNN layers
        activation = inputs_tr

        modules_list = [64, 64, 128, 128]
        for j in range(len(modules_list)):
            input_channels = activation.shape[-1]
            module_count = modules_list[j]
            filter_shape = [3, 3, input_channels, modules_list[j]]
            activation = modular.conv_layer(activation,
                                            filter_shape,
                                            strides=[1, 2, 2, 1],
                                            pool=False)

        flattened = tf.layers.flatten(activation)

        modules_list = [2, 1]
        units = 192
        for i in range(len(modules_list)):
            flattened = tf.layers.dense(flattened, modules_list[i] * units,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            flattened = modular.batch_norm(flattened)

        logits = tf.layers.dense(flattened, units=10)

        target = labels_cast
        loglikelihood = tf.reduce_mean(tf.distributions.Categorical(logits).log_prob(target))

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        return (loglikelihood, accuracy)

    template = tf.make_template('network', network)

    (ll,
     accuracy) = template()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-ll)

    create_summary(tf.reduce_mean(ll), 'loglikelihood', 'scalar')
    create_summary(accuracy, 'accuracy', 'scalar')

    with tf.Session() as sess:
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())

        if REALRUN=='True':
            test_writer = tf.summary.FileWriter(
                f'logs/test:Baseline_Advanced_CNN_tutorial_no_pool_{time}',
                sess.graph)
            writer = tf.summary.FileWriter(
                f'logs/train:Baseline_Advanced_CNN_tutorial_no_pool_{time}',
                sess.graph)

        general_summaries = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}


        for i in tqdm(range(100000)):

            # Sometimes generate summaries
            if i % 50 == 0:
                summaries = general_summaries
                _, summary_data = sess.run(
                    [opt, summaries],
                    train_dict)

                if REALRUN=='True':
                    writer.add_summary(summary_data, global_step=i)

                    summary_data = sess.run(summaries, test_dict)
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
                sess.run(opt, train_dict)

        if REALRUN == 'True':
            writer.close()
            test_writer.close()


if __name__ == '__main__':
    run()
