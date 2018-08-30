from libmodular.layers import create_ema_opt, get_sparsity_level, get_dep_pi_level
from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES, get_tensor_op, get_op, get_KL
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import datetime
import libmodular as modular
import numpy as np
import observations
import os
import random
import sys
import tensorflow as tf

cwd = os.getcwd()

REALRUN = sys.argv[1]
E_step = sys.argv[2]
masked_bernoulli = sys.argv[3]
variational = sys.argv[4]
beta_bern = sys.argv[5]
beta = 1

def create_sparse_summary(sparse_ops):
    def layer_sparsity(op):
        batch_sparse = tf.reduce_sum(op, axis=1)/tf.cast((tf.shape(op)[1]), tf.float32)
        return tf.reduce_mean(batch_sparse)
    sparse_model = tf.reduce_mean([layer_sparsity(op) for op in sparse_ops ])
    create_summary(sparse_model, 'Sparsity ratio', 'scalar')

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

def sum_and_mean_il(il, sample_size, tile_shape):
    il = tf.reshape(il, [tile_shape,
                        sample_size])
    il = tf.reduce_sum(il, axis=0)
    return tf.reduce_mean(il, axis=0)

# noinspection PyProtectedMember
def run():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = observations.cifar10('~/data/cifar10')
    y_test = y_test.astype(np.uint8)  # Fix test_data dtype


    x_train, y_train = x_train[0:100,:,:,:], y_train[0:100]

    dataset_size = x_train.shape[0]

    batch_size = 100
    num_batches = dataset_size/batch_size

    # Train dataset
    train = get_dataset(x_train, y_train, batch_size)

    # Test dataset
    test_batch_size = 50
    test = get_dataset(x_test, y_test, test_batch_size)

    # Handle to switch between datasets
    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(
        handle, train.output_types, train.output_shapes)
    data_indices, (inputs, labels) = itr.get_next()

    # Preprocessing
    inputs_cast = tf.cast(inputs, tf.float32) / 255.0
    inputs_tr = tf.transpose(inputs_cast, perm=(0, 2, 3, 1))
    labels_cast = tf.cast(labels, tf.int32)

    masked_bernoulli = False
    sample_size = 2
    epoch_lim = 15.

    iteration_number = tf.placeholder(dtype=tf.float32,
                                shape=[],
                                name='iteration_number')
    iteration = tf.placeholder(dtype=tf.float32,
                                shape=[],
                                name='iteration')

    def network(context: modular.ModularContext, 
                masked_bernoulli=False, 
                variational=False):
        # 4 modular CNN layers
        activation = inputs_tr
        s_log = []
        ctrl_logits =[]
        l_out_log = []
        pi_log = []
        bs_perst_log = []

        modules_list = [64, 64, 128, 128, 256, 256]
        for j in range(len(modules_list)):
            input_channels = activation.shape[-1]
            module_count = modules_list[j]
            filter_shape = [3, 3, input_channels, 1]
            modules = modular.create_conv_modules(filter_shape, 
                                                  module_count, 
                                                  strides=[1, 1, 1, 1])

            if masked_bernoulli == 'True' :
                print('Masked Bernoulli')
                hidden, l, s  = modular.masked_layer(
                    activation, modules, context, 
                    get_initialiser(dataset_size, 5, module_count))

            elif variational == 'True':
                print('Variational')
                hidden, l, s, pi, bs = modular.dep_variational_mask(
                    activation, modules, context, 0.001, tf.shape(inputs_tr)[0], iteration)
                hidden = modular.batch_norm(hidden)

            elif beta_bern == 'True':
                print('beta')
                hidden, l, s, pi, bs = modular.beta_bernoulli(
                    activation, modules, context, 0.001, tf.shape(inputs_tr)[0])

            else:
                print('Vanilla')
                hidden, l, s  = modular.modular_layer(
                    activation, modules, 3, context)
            ctrl_logits.append(tf.cast(tf.reshape(l, [1,-1,module_count,1]), tf.float32))
            s_log.append(tf.cast(tf.reshape(s, [1,-1,module_count,1]), tf.float32))
            pi_log.append(pi)
            bs_perst_log.append(tf.cast(tf.reshape(bs, [1,-1,module_count,1]), tf.float32))
            pooled = tf.nn.max_pool(
                hidden, 
                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                padding='SAME')
            activation = tf.nn.relu(pooled)

        flattened = tf.layers.flatten(activation)

        modules_list = [8]
        for i in range(len(modules_list)):
            module_count = modules_list[i]
            modules = modular.create_dense_modules(
                flattened, module_count,
                units=8, activation=tf.nn.relu)
            flattened, l, s, pi, bs = modular.dep_variational_mask(
                flattened, modules, context, 0.001,  tf.shape(inputs_tr)[0], iteration)
            flattened = modular.batch_norm(flattened)


            ctrl_logits.append(tf.cast(tf.reshape(l, [1,-1,module_count,1]), tf.float32))
            s_log.append(tf.cast(tf.reshape(s, [1,-1,module_count,1]), tf.float32))
            pi_log.append(pi)
            bs_perst_log.append(tf.cast(tf.reshape(bs, [1,-1,module_count,1]), tf.float32))

        logits = tf.layers.dense(flattened, units=10)

        target = modular.modularize_target(labels_cast, context)
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target)

        loglikelihood = sum_and_mean_il(loglikelihood, context.sample_size, tf.shape(inputs_tr)[0])

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), tf.float32))


        return (loglikelihood, logits, accuracy, 
                ctrl_logits, s_log, 
                context, pi_log, bs_perst_log)

    template = tf.make_template('network', network, masked_bernoulli=masked_bernoulli, 
                                variational=variational)



    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    if variational == 'False':
        e_step, m_step, eval = modular.modularize(template, optimizer, dataset_size,
                                                  data_indices, sample_size=10, 
                                                  variational=variational)
    else:
        m_step, eval = modular.modularize_variational(template, optimizer, dataset_size,
                                                  data_indices, variational, num_batches, 
                                                  beta, sample_size, iteration_number, epoch_lim)

    (ll, logits, 
    accuracy, 
    ctrl_logits, 
    s_log, 
    context, 
    pi_log, bs_perst_log) = eval

    #summaries
    params = context.layers
    a_list = [l.a for l in params]
    b_list = [l.b for l in params]
    eta_list = [l.eta for l in params]
    khi_list = [l.khi for l in params]
    gamma_list = [l.gamma for l in params]

    create_sparse_summary(get_sparsity_level())

    create_summary(a_list, 'a', 'histogram')
    create_summary(b_list, 'b', 'histogram')

    create_summary(pi_log, 'pi', 'histogram')
    create_summary(ctrl_logits, 'Controller_probs', 'image')
    create_summary(s_log, 'Selection', 'image')
    create_summary(bs_perst_log, 'Best_selection', 'image')

    create_summary(tf.reduce_mean(ll), 'loglikelihood', 'scalar')
    create_summary(accuracy, 'accuracy', 'scalar')

    create_summary(get_dep_pi_level(), 'dep_pi', 'histogram')


    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())

        if REALRUN=='True':
            test_writer = tf.summary.FileWriter(
                f'logs/test:Cifar10_variational_mask:a:3.5_b:0.5_alpha:0.01_samples:2_epochlim:10_anneal:5_Dependent_NOPI_for_infer_between:6000-15000_AND_30000-40000_filters:64,64,128,128,256,256_New_Init_{time}', sess.graph)
            writer = tf.summary.FileWriter(
                f'logs/train:Cifar10_variational_mask:a:3.5_b:0.5_alpha:0.01_samples:2_epochlim:10_anneal:5_Dependent_NOPI_for_infer_between:6000-15000_AND_30000-40000_filters:64,64,128,128,256,256_New_Init_{time}', sess.graph)

        general_summaries = tf.summary.merge_all()
        m_step_summaries = tf.summary.merge([create_m_step_summaries(), general_summaries])
        sess.run(tf.global_variables_initializer())
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}

        if E_step == 'True' and variational == 'False':
            print('EEEEE')
            for i in tqdm(range(200)):
                _ = sess.run(e_step, train_dict)

        j_s = 0.
        for i in tqdm(range(100000)):
            # Switch between E-step and M-step
            train_dict[iteration_number] = j_s
            test_dict[iteration_number] = j_s

            train_dict[iteration] = i+0.
            test_dict[iteration] = i+0.

            if variational or beta_bern == 'True':
                step = m_step
            else:
                step = e_step if i % 1 == 0 else m_step

            # Sometimes generate summaries
            if i % 50 == 0: 
                summaries = m_step_summaries
                _, summary_data, test_accuracy = sess.run(
                    [step, summaries, accuracy], 
                    train_dict)

                if REALRUN=='True':
                    writer.add_summary(summary_data, global_step=i) 

                    # summary_data = sess.run(summaries, test_dict)
                    # test_writer.add_summary(summary_data, global_step=i)

                    # accuracy_log = []
                    # for test in range(x_test.shape[0]//test_batch_size):
                    #     test_accuracy = sess.run(accuracy, test_dict)
                    #     accuracy_log.append(test_accuracy)
                    # final_accuracy = np.mean(accuracy_log)
                    # summary = tf.Summary()
                    # summary.value.add(tag='Test Accuracy', 
                    #                   simple_value = final_accuracy)
                    # test_writer.add_summary(summary, global_step=i)

            else:
                sess.run(step, train_dict)

            if i % (dataset_size//batch_size) == 0 and j_s<epoch_lim-1:
                j_s+=1.
            elif j_s>epoch_lim-1:
                j_s = epoch_lim-1
            else:
                j_s = j_s

        # if not os.path.exists(os.path.dirname(cwd + '/model')):
        #     os.makedirs(os.path.dirname(cwd + '/model'))
        # saver.save(sess, cwd + '/model')

        if REALRUN=='True':
            writer.close()
            test_writer.close()


if __name__ == '__main__':
    run()
