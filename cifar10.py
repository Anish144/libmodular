from libmodular.layers import create_ema_opt, get_sparsity_level, get_dep_input
from libmodular.modular import create_m_step_summaries, M_STEP_SUMMARIES
from libmodular.modular import get_tensor_op, get_op, get_KL
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

FLAGS = tf.app.flags.FLAGS

# Flags
tf.app.flags.DEFINE_boolean('real_run', True, """Save model information""")
tf.app.flags.DEFINE_string('name', 'Default_Run',
                           """Name to be saved under""")
tf.app.flags.DEFINE_integer('batch_size', 25,
                            """Batch size""")
tf.app.flags.DEFINE_integer('test_batch_size', 25,
                            """Test Batch size""")
tf.app.flags.DEFINE_integer('sample_size', 1,
                            """Sample size for estimator""")
tf.app.flags.DEFINE_integer('epoch_lim', 5,
                            """N_0 for annealing""")
tf.app.flags.DEFINE_integer('damp_length', 5,
                            """N_1 for annealing""")
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Alpha hyperparam for dropout prior""")
tf.app.flags.DEFINE_boolean('output_add', False,
                            """Average functionality of modules""")
tf.app.flags.DEFINE_boolean('cnn_ctrl', False,
                            """Convolutional controller""")
tf.app.flags.DEFINE_boolean('debug', False,
                            """Run only basic strucutre of model""")
tf.app.flags.DEFINE_integer('training_steps', 100000,
                            """How long to run experiment for""")

arguments = {
    'cnn_module_list': [8, 8, 8, 8
                        ],
    'cnn_filter_size': [8, 8, 16, 16],
    'linear_module_list': [8, 4],
    'linear_units': 48,
    'a_init_range': [3.5, 3.5],
    'b_init_range': [0.3, 0.3],
    'Datasets': ['cifar10'],
}


def fix_image_summary(list_op, op, module_count):
    list_op.append(
        tf.cast(
            tf.reshape(
                op,
                [1, -1, module_count, 1]),
            tf.float32))
    pass


def create_sparse_summary(sparse_ops):
    def layer_sparsity(op):
        layer_count = sparse_ops.index(op)
        batch_sparse = tf.reduce_sum(op, axis=1) / (tf.cast((tf.shape(op)[1]),
                                                            tf.float32))
        layer_sparse = tf.reduce_mean(batch_sparse)
        create_summary(
            layer_sparse, 'layer_{}_sparsity'.format(layer_count), 'scalar')
        return layer_sparse
    sparse_model = tf.reduce_mean([layer_sparsity(op) for op in sparse_ops])
    create_summary(sparse_model, 'Sparsity ratio', 'scalar')


def get_initialiser(data_size, n, module_count):
    choice = np.zeros((data_size, n), dtype=int)
    for j in range(data_size):
        choice[j, :] = random.sample(range(module_count), n)
    one_hot = np.zeros((data_size, module_count), dtype=int)
    for i in range(n):
        one_hot[np.arange(data_size), choice[:, i]] = 1
    return tf.constant_initializer(one_hot, dtype=tf.int32, verify_shape=True)


def make_handle(sess, dataset):
    iterator = dataset.make_initializable_iterator()
    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle


def get_dataset(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    prepare = data._enumerate().repeat().shuffle(50000)
    return prepare.batch(batch_size)


def create_summary(list_of_ops_or_op, name, summary_type):
    summary = getattr(tf.summary, summary_type)

    if type(list_of_ops_or_op) is list:
        for i in range(len(list_of_ops_or_op)):
            summary(str(name) + '_' + str(i), list_of_ops_or_op[i])

    elif type(list_of_ops_or_op) is tf.Tensor:
        summary(str(name), list_of_ops_or_op)

    elif list_of_ops_or_op is None:
        pass

    else:
        raise TypeError('Invalid type for summary')


def sum_and_mean_il(il, sample_size, tile_shape):
    il = tf.reshape(il, [tile_shape,
                    sample_size])
    il = tf.reduce_sum(il, axis=0)
    return tf.reduce_mean(il, axis=0)


# noinspection PyProtectedMember
def run():
    """
    Run the model on the specific dataset.

    No args.
    """

    x_train_full = []
    x_test_full = []
    y_train_full = []
    y_test_full = []

    # Load and Preprocess data
    for dataset in arguments['Datasets']:
        pull_dataset = getattr(observations, dataset)
        (x_train, y_train), (x_test, y_test) = pull_dataset(
            '~/data/{}'.format(dataset))
        if FLAGS.debug:
            x_train, y_train = x_train[:100, :, :, :], y_train[:100]
        y_test = y_test.astype(np.uint8)
        if dataset == 'cifar10':
            x_train = np.transpose(x_train, [0, 2, 3, 1])
            x_test = np.transpose(x_test, [0, 2, 3, 1])
        x_train_full.append(x_train)
        x_test_full.append(x_test)
        y_train_full.append(y_train)
        y_test_full.append(y_test)

    x_train = np.vstack(x_train_full)
    x_test = np.vstack(x_test_full)
    y_train = np.vstack(y_train_full)[0, :]
    y_test = np.vstack(y_test_full)[0, :]

    dataset_size = x_train.shape[0]

    num_batches = dataset_size / FLAGS.batch_size

    # Train dataset
    train = get_dataset(x_train, y_train, FLAGS.batch_size)

    # Test dataset
    test = get_dataset(x_test, y_test, FLAGS.test_batch_size)

    # Handle to switch between datasets
    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(
        handle, train.output_types, train.output_shapes)
    data_indices, (inputs, labels) = itr.get_next()

    inputs_tr = tf.cast(inputs, tf.float32) / 255.0
    labels_cast = tf.cast(labels, tf.int32)

    iteration_number = tf.placeholder(
        dtype=tf.float32,
        shape=[],
        name='iteration_number')
    iterate = tf.placeholder(
        dtype=tf.float32,
        shape=[],
        name='iterate')

    def network(context: modular.ModularContext):
        # 4 modular CNN layers
        activation = inputs_tr
        s_log = []
        ctrl_logits = []
        l_out_log = []
        pi_log = []
        bs_perst_log = []

        # CNN layers
        for j in range(len(arguments['cnn_module_list'])):
            input_channels = activation.shape[-1]
            module_count = arguments['cnn_module_list'][j]
            out_channel = arguments['cnn_filter_size'][j]
            filter_shape = [3, 3, input_channels, out_channel]
            modules = modular.create_conv_modules(filter_shape,
                                                  module_count,
                                                  strides=[1, 2, 2, 1])

            hidden, l, s, pi, bs = modular.dep_variational_mask(
                inputs=activation,
                modules=modules,
                context=context,
                tile_shape=tf.shape(inputs_tr)[0],
                iteration=iterate,
                a_init=arguments['a_init_range'],
                b_init=arguments['b_init_range'],
                output_add=FLAGS.output_add,
                cnn_ctrl=FLAGS.cnn_ctrl)

            fix_image_summary(ctrl_logits, l, module_count)
            fix_image_summary(s_log, s, module_count)
            fix_image_summary(bs_perst_log, bs, module_count)
            pi_log.append(pi)

            activation = modular.batch_norm(hidden)

        flattened = tf.layers.flatten(activation)

        # Linear layers
        for i in range(len(arguments['linear_module_list'])):
            print('Linear')
            module_count = arguments['linear_module_list'][i]
            modules = modular.create_dense_modules(
                flattened,
                module_count,
                units=arguments['linear_units'])
            flattened, l, s, pi, bs = modular.dep_variational_mask(
                inputs=flattened,
                modules=modules,
                context=context,
                tile_shape=tf.shape(inputs_tr)[0],
                iteration=iterate,
                a_init=arguments['a_init_range'],
                b_init=arguments['b_init_range'],
                output_add=FLAGS.output_add,
                cnn_ctrl=False)

            flattened = modular.batch_norm(flattened)

            fix_image_summary(ctrl_logits, l, module_count)
            fix_image_summary(s_log, s, module_count)
            fix_image_summary(bs_perst_log, bs, module_count)
            pi_log.append(pi)

        logits = tf.layers.dense(flattened, units=10)

        target = modular.modularize_target(labels_cast, context)
        loglikelihood = tf.distributions.Categorical(logits).log_prob(target)

        loglikelihood = sum_and_mean_il(loglikelihood,
                                        context.sample_size,
                                        tf.shape(inputs_tr)[0])

        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    predicted,
                    target),
                tf.float32))

        return (loglikelihood, logits, accuracy,
                ctrl_logits, s_log,
                context, pi_log, bs_perst_log)

    template = tf.make_template(
        'network',
        network)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    m_step, eval = modular.modularize_variational(
        template,
        optimizer,
        dataset_size,
        data_indices,
        num_batches,
        FLAGS.sample_size,
        iteration_number,
        FLAGS.epoch_lim,
        FLAGS.damp_length,
        FLAGS.alpha)

    (ll,
        logits,
        accuracy,
        ctrl_logits,
        s_log,
        context,
        pi_log, bs_perst_log) = eval

    # summaries
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
    create_summary(ctrl_logits, 'Ctrl_and_pi', 'image')
    create_summary(s_log, 'Inference', 'image')
    create_summary(bs_perst_log, 'Ctrl', 'image')

    create_summary(tf.reduce_mean(ll), 'loglikelihood', 'scalar')
    create_summary(accuracy, 'accuracy', 'scalar')

    create_summary(get_dep_input(), 'dep_input', 'histogram')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if FLAGS.debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())

        if FLAGS.real_run:
            test_writer = tf.summary.FileWriter(
                (f'logs/test:'
                 'name:{}_'.format(FLAGS.name) +
                 'alpha:{}_'.format(FLAGS.alpha) +
                 'samples:.{}_'.format(FLAGS.sample_size) +
                 'epoch_lim:{}_'.format(FLAGS.epoch_lim) +
                 'damp_length:{}_'.format(FLAGS.damp_length) +
                 'a:{}_'.format(arguments['a_init_range']) +
                 'b:{}_'.format(arguments['b_init_range']) +
                 # 'module_list:{}_'.format(arguments['cnn_module_list']) +
                 # 'filter_size:{}_'.format(arguments['cnn_filter_size']) +
                 'output_add:{}_'.format(FLAGS.output_add) +
                 'cnn_ctrl:{}_'.format(FLAGS.cnn_ctrl) +
                 f'{time}'), sess.graph)

            writer = tf.summary.FileWriter(
                (f'logs/train:'
                 'name:{}_'.format(FLAGS.name) +
                 'alpha:{}_'.format(FLAGS.alpha) +
                 'samples:.{}_'.format(FLAGS.sample_size) +
                 'epoch_lim:{}_'.format(FLAGS.epoch_lim) +
                 'damp_length:{}_'.format(FLAGS.damp_length) +
                 'a:{}_'.format(arguments['a_init_range']) +
                 'b:{}_'.format(arguments['b_init_range']) +
                 # 'module_list:{}_'.format(arguments['cnn_module_list']) +
                 # 'filter_size:{}_'.format(arguments['cnn_filter_size']) +
                 'output_add:{}_'.format(FLAGS.output_add) +
                 'cnn_ctrl:{}_'.format(FLAGS.cnn_ctrl) +
                 f'{time}'), sess.graph)

        general_summaries = tf.summary.merge_all()
        m_step_summaries = tf.summary.merge(
            [create_m_step_summaries(),
                general_summaries])
        sess.run(tf.global_variables_initializer())
        train_dict = {handle: make_handle(sess, train)}
        test_dict = {handle: make_handle(sess, test)}

        j_s = 0.
        for i in tqdm(range(FLAGS.training_steps)):
            train_dict[iteration_number] = j_s
            test_dict[iteration_number] = j_s

            train_dict[iterate] = i
            test_dict[iterate] = i

            step = m_step

            # Sometimes generate summaries
            if i % 50 == 0:
                summaries = m_step_summaries
                _, summary_data, test_accuracy = sess.run(
                    [step, summaries, accuracy],
                    train_dict)

                if FLAGS.real_run:
                    writer.add_summary(summary_data, global_step=i)

                    summary_data = sess.run(summaries, test_dict)
                    test_writer.add_summary(summary_data, global_step=i)

                    accuracy_log = []
                    for test in range(
                            x_test.shape[0] // FLAGS.test_batch_size):
                        test_accuracy = sess.run(accuracy, test_dict)
                        accuracy_log.append(test_accuracy)
                    final_accuracy = np.mean(accuracy_log)
                    summary = tf.Summary()
                    summary.value.add(tag='Test Accuracy',
                                      simple_value=final_accuracy)
                    test_writer.add_summary(summary, global_step=i)

            else:
                _, da = sess.run([step, get_op()], train_dict)
                print('Damp:', da)
                print('iteration:', j_s)

            warm_up = FLAGS.epoch_lim + FLAGS.damp_length
            if (i % (dataset_size // FLAGS.batch_size) == 0 and
               j_s < warm_up - 1):
                j_s += 1.
            elif j_s >= warm_up - 1:
                j_s = warm_up - 1

            if FLAGS.real_run:
                if i % 5000 == 0:
                    save = saver.save(
                        sess, cwd + "/tmp/model.ckpt", global_step=i)

        if FLAGS.real_run:
            writer.close()
            test_writer.close()


if __name__ == '__main__':
    run()
