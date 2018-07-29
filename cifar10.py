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
from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES

REALRUN = sys.argv[1]
E_step = sys.argv[2]
new_controller = sys.argv[3]
variational = sys.argv[4]

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

    batch_size = 250
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

    module_count = 16
    masked_bernoulli = False

    def network(context: modular.ModularContext, masked_bernoulli=False, variational=False):
        # 4 modular CNN layers
        activation = inputs_tr
        bs_log = []
        ctrl_logits =[]
        l_out_log = []
        ema_list = []

        input_channels = activation.shape[-1]
        filter_shape = [3, 3, input_channels, 8]
        activation = modular.conv_layer(activation, filter_shape, strides=[1,1,1,1])


        # input_channels = activation.shape[-1]
        # filter_shape = [3, 3, input_channels, 16]
        # activation = modular.conv_layer(activation, filter_shape, strides=[1,1,1,1])


        # input_channels = activation.shape[-1]
        # filter_shape = [3, 3, input_channels, 32]
        # activation = modular.conv_layer(activation, filter_shape, strides=[1,1,1,1])

        for j in range(2):
            input_channels = activation.shape[-1]
            filter_shape = [3, 3, input_channels, 8]
            modules = modular.create_conv_modules(filter_shape, module_count, strides=[1, 1, 1, 1])
            if masked_bernoulli:
                print('Maksed Bernoulli')
                hidden, l, bs  = modular.masked_layer(activation, modules, context,
                                                     get_initialiser(dataset_size, 5, module_count))
            elif variational == 'True':
                print('Variational')
                hidden, l, bs, l_out = modular.variational_mask(activation, modules, context, 0.001, 3.17)
                hidden = modular.batch_norm(hidden)
            elif new_controller == 'True':
                print('New')
                hidden, ema_out, l, bs = modular.new_controller(activation, modules, context, 
                                                      get_initialiser(dataset_size, 5, module_count))
                hidden = modular.batch_norm(hidden)
            else:
                print('Vanilla')
                hidden, l, bs  = modular.modular_layer(activation, modules, 3, context)
            ctrl_logits.append(l)
            bs_log.append(bs)
            ema_list.append(ema_out)
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

        return (loglikelihood, logits, accuracy, selection_entropy, batch_selection_entropy, 
                tf.group(ema_list), ctrl_logits, bs_log, l_out_log)

    template = tf.make_template('network', network, masked_bernoulli=masked_bernoulli, 
                                variational=variational)

    ll, logits, accuracy, s_entropy, bs_entropy, ema_opt, ctrl_logits, bs_log, l_out_log = modular.evaluation(template,
                                                                data_indices,
                                                                dataset_size)

    with tf.control_dependencies([ema_opt]):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)

    if variational == 'False':
        e_step, m_step = modular.modularize(template, optimizer, dataset_size,
                                                  data_indices, sample_size=10, 
                                                  variational=variational, 
                                                  moving_average=ema_opt)
    else:
        m_step, eval = modular.modularize_variational(template, optimizer, dataset_size,
                                                  data_indices, variational)


    for i in range(len(ctrl_logits)):
        l = tf.reshape(ctrl_logits[i], [1,-1,module_count,1])
        tf.summary.image('l' + str(i) + '_controller_probs', l, max_outputs=1)

    for i in range(len(bs_log)):
        bs = tf.reshape(bs_log[i] , [1,-1,module_count,1])
        tf.summary.image('best_selection_'+ str(i), tf.cast(bs, dtype=tf.float32), max_outputs=1)



    tf.summary.scalar('loglikelihood', tf.reduce_mean(ll))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('entropy/exp_selection', tf.exp(s_entropy))
    tf.summary.scalar('entropy/exp_batch_selection', tf.exp(bs_entropy))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())

        if REALRUN=='True':
            writer = tf.summary.FileWriter(f'logs/train:Cifar10_16m_ADDED_Moving_average:0.5_EVERYHTINGREDUCED:alpha:0.1_Initial_a=3.8-4.2,b=1.8-2.2_lr:0.005:'+f'_{time}', sess.graph)
            test_writer = tf.summary.FileWriter(f'logs/test:Cifar10_16m_ADDED_Moving_average:0.5_EVERYHTINGREDUCED:alpha:0.1_Initial_a=3.8-4.2,b=1.8-2.2_lr:0.005:'+f'_{time}', sess.graph)
            # writer = tf.summary.FileWriter(f'logs/train:Cifar10_Variational_with_everything_reducemeaned_alpha=1.9_a=3.8-4.2,b=1.8-2.2_{time}')
            # test_writer = tf.summary.FileWriter(f'logs/test:Cifar10_Variational_with_everything_reducemeaned_alpha=1.9_a=3.8-4.2,b=1.8-2.2_{time}')

        general_summaries = tf.summary.merge_all()
        m_step_summaries = tf.summary.merge([create_m_step_summaries(), general_summaries])
        sess.run(tf.global_variables_initializer())
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}

        if E_step == 'True' and variational == 'False':
            print('EEEEE')
            for i in tqdm(range(200)):
                _ = sess.run(e_step, train_dict)

        for i in tqdm(range(400000)):
            # Switch between E-step and M-step
            if variational == 'True':
                step = m_step
            else:  
                step = e_step if i % 30 == 0 else m_step


            # Sometimes generate summaries
            if i % 100 == 0:
                summaries = m_step_summaries
                _, summary_data, test_accuracy, log, bs = sess.run([step, summaries, accuracy, 
                                                                       ctrl_logits[0], bs_log[0], 
                                                                       ], train_dict)
                # non_zeros = np.zeros((log.shape[0], log.shape[1]))
                # j=0
                # for i in log:
                #     zeros = np.zeros((len(i)))
                #     zeros[i>0.2] = 1
                #     non_zeros[j,:] = zeros
                #     j+=1
                # plt.bar(np.arange(len(i)), np.sum(non_zeros, axis=0))
                # plt.savefig('Sparsemaxtest.png')
                # import pdb; pdb.set_trace()

                if REALRUN=='True':
                    writer.add_summary(summary_data, global_step=i) 

                summary_data = sess.run(general_summaries, test_dict)

                if REALRUN=='True':
                    test_writer.add_summary(summary_data, global_step=i)

            else:
                sess.run(step, train_dict)

        if REALRUN=='True':
            writer.close()
            test_writer.close()


if __name__ == '__main__':
    run()
