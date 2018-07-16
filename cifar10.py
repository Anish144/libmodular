import datetime
import random
import tensorflow as tf
import numpy as np
import libmodular as modular
import observations
from tqdm import tqdm
import sys

from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES

E_step = sys.argv[1]
new_controller = sys.argv[2]


def get_initialiser(data_size, n, module_count):
    choice = np.zeros((data_size, n), dtype=int)
    for j in range(data_size):
        choice[j,:] = random.sample(range(module_count), n)
    one_hot = np.zeros((data_size, module_count), dtype=int)
    for i in range(n):
        one_hot[np.arange(data_size), choice[:,i]]=1
    return tf.constant_initializer(one_hot, dtype=tf.int32, verify_shape=True)

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

    batch_size = 125
    # Train dataset
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))._enumerate().repeat().shuffle(50000).batch(batch_size)
    # Test dataset
    # dummy_data_indices = tf.zeros([test_batch_size], dtype=tf.int64)
    test_batch_size = 2000
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))._enumerate().repeat().batch(test_batch_size)

    # Handle to switch between datasets
    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(handle, train.output_types, train.output_shapes)
    data_indices, (inputs, labels) = itr.get_next()

    # Preprocessing
    inputs_cast = tf.cast(inputs, tf.float32) / 255.0
    inputs_tr = tf.transpose(inputs_cast, perm=(0, 2, 3, 1))
    labels_cast = tf.cast(labels, tf.int32)

    module_count = 10
    variational = False
    masked_bernoulli = False

    def network(context: modular.ModularContext, masked_bernoulli=False, variational=False):
        # 4 modular CNN layers
        activation = inputs_tr
        ctrl_logits=[]
        for j in range(4):
            input_channels = activation.shape[-1]
            filter_shape = [3, 3, input_channels, 8]
            modules = modular.create_conv_modules(filter_shape, module_count, strides=[1, 1, 1, 1])
            if masked_bernoulli:
                print('WORKS')
                hidden, l, bs  = modular.masked_layer(activation, modules, context, get_initialiser(dataset_size, 5, module_count))
            elif variational:
                print('Doesnt work')
                hidden = modular.variational_mask(activation, modules, context, 0.001, 3.17, get_initialiser(dataset_size, 5, module_count))
            elif new_controller == 'True':
                print('NEW')
                hidden, l, bs = modular.new_controller(activation, modules, context,  get_initialiser(dataset_size, 5, module_count))
            else:
                print('UNMASKED')
                hidden, l, bs  = modular.modular_layer(activation, modules, 3, context)
            ctrl_logits.append(l)
            pooled = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            activation = tf.nn.relu(pooled)

        flattened = tf.layers.flatten(activation)
        logits = tf.layers.dense(flattened, units=10)

        target = modular.modularize_target(labels_cast, context)
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target)

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))

        selection_entropy = context.selection_entropy()
        batch_selection_entropy = context.batch_selection_entropy()

        return loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy, ctrl_logits, bs

    template = tf.make_template('network', network, masked_bernoulli=masked_bernoulli, variational=variational)
    optimizer = tf.train.AdamOptimizer()

    if not variational:
        e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                                  data_indices, sample_size=15, variational=variational)
    else:
        m_step, eval = modular.modularize_variational(template, optimizer, dataset_size,
                                                  data_indices, variational)
    ll, logits, accuracy, s_entropy, bs_entropy, ctrl_logits, bs = eval

    l1 = tf.reshape(ctrl_logits[0], [1,-1,module_count,1])
    l2 = tf.reshape(ctrl_logits[1], [1,-1,module_count,1])
    l3 = tf.reshape(ctrl_logits[2], [1,-1,module_count,1])

    tf.summary.image('l1_controller_probs', l1, max_outputs=1)
    tf.summary.image('l2_controller_probs', l2, max_outputs=1)
    tf.summary.image('l3_controller_probs', l3, max_outputs=1)

    bs = tf.reshape(bs, [1,-1,module_count,1])

    tf.summary.image('bs_persistent',  tf.cast(bs, dtype=tf.float32), max_outputs=1)

    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        # writer = tf.summary.FileWriter(f'logs/train_10m_maskedBernoulli_TRIAL_NEWINIT:Loop_E:' + str(E_step)+ f'_{time}', sess.graph)
        # test_writer = tf.summary.FileWriter(f'logs/test_10m_maskedBernoulli_TRIAL_NEWINIT:Loop_E:' + str(E_step)+ f'_{time}', sess.graph)
        writer = tf.summary.FileWriter(f'logs/train:New_Controller_CIFAR10_:'+f'_{time}', sess.graph)
        test_writer = tf.summary.FileWriter(f'logs/test:New_Controller_CIFAR10_:'+f'_{time}', sess.graph)
        general_summaries = tf.summary.merge_all()
        m_step_summaries = tf.summary.merge([create_m_step_summaries(), general_summaries])
        sess.run(tf.global_variables_initializer())
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}


        if E_step == 'True':
            print('EEEEE')
            for i in tqdm(range(400)):
                _ = sess.run(e_step, train_dict)

        for i in tqdm(range(600000)):
            # Switch between E-step and M-step
            if variational:
                step = m_step
            else:
                step = e_step if i % 30 == 0 else m_step

            # Sometimes generate summaries
            if i % 25 == 0:
                summaries = m_step_summaries if step == m_step else general_summaries
                _, summary_data = sess.run([step, summaries], train_dict)
                writer.add_summary(summary_data, global_step=i) 
                summary_data = sess.run(general_summaries, test_dict)
                test_writer.add_summary(summary_data, global_step=i)
                writer.flush()
                test_writer.flush()
            else:
                sess.run(step, train_dict)

        writer.close()
        test_writer.close()


if __name__ == '__main__':
    run()
