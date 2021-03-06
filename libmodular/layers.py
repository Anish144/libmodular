import tensorflow as tf
from tensorflow.contrib import distributions as tfd
import numpy as np

from libmodular.modular import ModulePool, ModularContext, ModularMode
from libmodular.modular import ModularLayerAttributes
from libmodular.modular import run_modules, run_masked_modules
from libmodular.modular import e_step, m_step, evaluation
from libmodular.modular import run_masked_modules_withloop
from libmodular.modular import run_modules_withloop
from libmodular.modular import run_masked_modules_withloop_and_concat
from libmodular.modular import run_non_modular
from tensorflow.python import debug as tf_debug
import os


def create_dense_modules(
    inputs_or_shape,
    module_count: int,
    units: int = None,
    activation=None
):
    """
    Takes in input, module count, units, and activation and returns a named
    tuple with a function
    that returns the multiplcation of a sepcific module with the input
    """
    with tf.variable_scope(None, 'dense_modules'):
        # Checks if input has attribute shape and takes last
        if hasattr(inputs_or_shape, 'shape') and units is not None:
            # First dimension is the weights of a specific module
            weights_shape = [module_count,
                             inputs_or_shape.shape[-1].value,
                             units]
        else:
            weights_shape = [module_count] + inputs_or_shape
        weights = tf.get_variable(
            'weights',
            weights_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        biases_shape = [module_count, units]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a, mask, w, b):
            """
            Takes in input and a module, multiplies input with
            the weights of the module
            weights are [module x input_shape x units]
            """
            # Put modules in the lef tmost axis
            with tf.variable_scope(None, "linear_loop"):
                w = tf.transpose(w, [1, 0, 2, 3])
                b = tf.transpose(b, [1, 0, 2])
                out = tf.einsum('bi,bio->bo', x, w[a]) + b[a]
                if activation is not None:
                    out = activation(out)
                return out

        return ModulePool(
            module_count,
            module_fnc,
            output_shape=[units], units=units,
            weight=weights, bias=biases)


def conv_layer(x, shape, strides, padding='SAME', pool=False):
    with tf.variable_scope(None, 'simple_conv_layer'):
        filter_shape = list(shape)
        filter = tf.get_variable(
            'filter',
            filter_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        biases_shape = [shape[-1]]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())
        hidden = tf.nn.conv2d(x, filter, strides, padding) + biases
        if pool:
            hidden = tf.nn.max_pool(
                hidden,
                ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        relu = tf.nn.relu(hidden)
        return batch_norm(relu)


def create_conv_modules(shape, module_count: int, strides, padding='SAME'):
    with tf.variable_scope(None, 'conv_modules'):
        filter_shape = [module_count] + list(shape)
        filter = tf.get_variable(
            'filter',
            filter_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        biases_shape = [module_count, shape[-1]]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a, mask, weight, bias):
            with tf.variable_scope(None, "depthwise_conv2d"):
                fw, fh, d = shape[0], shape[1], shape[-1]
                h, w, c = x.shape[1].value, x.shape[2].value, x.shape[-1].value
                b = tf.shape(x)[0]
                new_filter = tf.transpose(weight,
                                          [1, 0, 2, 3, 4, 5])
                new_biases = tf.transpose(bias,
                                          [1, 0, 2])
                new_biases = new_biases[a]
                new_filter = new_filter[a]
                # [fh,fw,b,c,d]
                new_filter = tf.transpose(new_filter, [1, 2, 0, 3, 4])
                # [fh,fw,b*c,d]
                new_filter = tf.reshape(new_filter,
                                        [fw, fh, b * c, d])
                # [h,w,b,c]
                x = tf.transpose(x, [1, 2, 0, 3])
                # [1,h,w,b*c]
                x = tf.reshape(x,
                               [1, h, w, b * c])

                out = tf.nn.depthwise_conv2d(
                    x,
                    filter=new_filter,
                    strides=strides,
                    padding=padding)
                if padding == "SAME":
                    out = tf.reshape(out, [h, w, -1, c, d])
                if padding == "VALID":
                    out = tf.reshape(out, [h - fh + 1, w - fw + 1, -1, c, d])
                out = tf.transpose(out, [2, 0, 1, 3, 4])
                out = tf.reduce_sum(out, axis=3)
                out = tf.transpose(out, [1, 2, 0, 3]) + new_biases
                out = tf.transpose(out, [2, 0, 1, 3])

                return out

        def module_fnc_original(x, a, mask, weight, bias):
            return tf.nn.conv2d(x, weight[a], strides, padding) + bias[a]

        def module_fnc_non_modular(x, a, mask, weight, bias):
            new_filter = tf.transpose(weight,
                                      [1, 0, 2, 3, 4, 5])
            new_biases = tf.transpose(bias,
                                      [1, 0, 2])
            new_biases = new_biases[a]
            new_filter = new_filter[a]
            return tf.nn.conv2d(
                x, new_filter, strides, padding) + new_biases[a]

        return ModulePool(module_count, module_fnc, output_shape=None,
                          units=list(shape)[-1], weight=filter, bias=biases)


def batch_norm(inputs):
    with tf.variable_scope(None, 'batch_norm'):
        return tf.layers.batch_normalization(inputs)


def dep_variational_mask(
    inputs, modules: ModulePool,
    context: ModularContext, tile_shape, iteration,
    a_init, b_init, output_add, cnn_ctrl
):
    """
    Constructs a Bernoulli masked layer that outputs sparse
    binary masks dependent on the input
    Based on the Adaptive network Sparsification paper
    Args:
        inputs; Batch X dimension
        modules; ModulePool named tuple
        context; context class
        eps; threshold for dependent pi
    """
    with tf.variable_scope(None, 'dep_variational_mask'):
        inputs = context.begin_modular(inputs)

        new_inputs = tf.stop_gradient(inputs)

        if output_add:
            function = run_masked_modules_withloop
        else:
            function = run_masked_modules_withloop_and_concat

        shape = modules.module_count
        u_shape = [context.sample_size, shape]

        a = tf.maximum(tf.get_variable(name='a',
                       dtype=tf.float32,
                       initializer=tf.random_uniform(
                           [shape], minval=a_init[0],
                           maxval=a_init[1])), 1e-20)
        b = tf.maximum(tf.get_variable(name='b',
                       dtype=tf.float32,
                       initializer=tf.random_uniform(
                           [shape], minval=b_init[0],
                           maxval=b_init[1])), 1e-20)

        pi = get_pi(a, b, u_shape)
        pi_batch = tf.expand_dims(pi, 1)
        pi_batch = tf.tile(
            pi_batch, [1, tile_shape, 1])
        pi_batch = tf.reshape(
            pi_batch,
            [tile_shape * context.sample_size, shape])

        initializer = tf.contrib.layers.xavier_initializer()

        def dependent_pi(ctrl_inputs, pi, cnn_ctrl):
            with tf.variable_scope('dep_pi', reuse=tf.AUTO_REUSE):
                if cnn_ctrl:
                    ctrl_output = tf.layers.conv2d(
                        inputs=ctrl_inputs,
                        filters=2,
                        kernel_size=[3, 3],
                        padding="same")
                    ctrl_inputs = ctrl_output

                flat_inputs = tf.layers.flatten(ctrl_inputs)
                W = tf.get_variable(
                    name='ctrl_weights',
                    shape=[flat_inputs.shape[-1].value, shape],
                    initializer=initializer,
                    trainable=True)
                b = tf.get_variable(
                    name='ctrl_bias',
                    shape=[shape],
                    initializer=tf.ones_initializer(),
                    trainable=True)
                dep_input = tf.sigmoid(tf.matmul(flat_inputs, W) + b)

                tf.add_to_collection(
                    name='ctrl_weights',
                    value=W)
                tf.add_to_collection(
                    name='ctrl_bias',
                    value=b)

                return tf.multiply(dep_input, pi), dep_input

        dep_pi = tf.make_template('dependent_pi', dependent_pi)

        new_pi, dep_input = dep_pi(new_inputs, pi_batch, cnn_ctrl)

        tf.add_to_collection(
            name='dep_input',
            value=dep_input)

        tau = 0.01
        z = relaxed_bern(
            tau, new_pi, [tile_shape * context.sample_size, shape])

        if context.mode == ModularMode.M_STEP:
            test_pi = new_pi
            selection = tf.round(z)
            final_selection = selection

        elif context.mode == ModularMode.EVALUATION:
            test_pi = get_test_pi(a, b)

            def before_cond():
                new_pi, dep_input_sample = dep_pi(
                    new_inputs, test_pi, cnn_ctrl)
                selection = tf.where(new_pi > 0.5,
                                     x=tf.ones_like(new_pi),
                                     y=tf.zeros_like(new_pi)
                                     )
                final_selection = selection
                return final_selection, selection

            def after_ind():
                selection = tf.where(test_pi > 0.5,
                                     x=tf.ones_like(test_pi),
                                     y=tf.zeros_like(test_pi)
                                     )
                final_selection = selection
                final_selection = tf.tile(
                    selection, [tf.shape(new_inputs)[0]])
                final_selection = tf.reshape(
                    final_selection, [tf.shape(new_inputs)[0], shape])
                return final_selection, selection

            great_1 = tf.greater(iteration, tf.constant(6000.))
            less_1 = tf.less(iteration, tf.constant(10000.))
            cond_1 = tf.logical_and(great_1, less_1)

            great_2 = tf.greater(iteration, tf.constant(15000.))
            less_2 = tf.less(iteration, tf.constant(20000.))
            cond_2 = tf.logical_and(great_2, less_2)

            great_3 = tf.greater(iteration, tf.constant(35000.))
            less_3 = tf.less(iteration, tf.constant(45000.))
            cond_3 = tf.logical_and(great_3, less_3)

            cond = tf.logical_or(tf.logical_or(cond_1, cond_2), cond_3)

            final_selection, selection = tf.cond(cond,
                                                 after_ind,
                                                 before_cond)

            tf.add_to_collection(
                name='sparsity',
                value=final_selection)

        attrs = ModularLayerAttributes(
            selection,
            None, None,
            a, b, pi, None, None,
            None, None, None)
        context.layers.append(attrs)

        if inputs.shape.ndims > 3:
            new_biases = tf.einsum(
                'mo,bm->bmo',
                modules.bias,
                z)
            new_weights = tf.einsum(
                'miocd,bm->bmiocd',
                modules.weight,
                z)
        else:
            new_weights = tf.einsum(
                'mio,bm->bmio',
                modules.weight,
                tf.cast(z, tf.float32))
            new_biases = tf.einsum(
                'mo,bm->bmo',
                modules.bias,
                tf.cast(z, tf.float32))

        return (function(inputs,
                final_selection,
                z,
                shape,
                modules.units,
                modules.module_fnc,
                modules.output_shape,
                new_weights,
                new_biases),
                new_pi, selection, test_pi, dep_input)


def get_test_pi(a, b):
    with tf.variable_scope('test_pi'):
        max_a = tf.check_numerics(tf.maximum(a, 1e-20), 'a is going NaN')
        div_a = tf.check_numerics(tf.realdiv(1., max_a), 'div_a')
        denom = tf.check_numerics(
            tf.lgamma(1 + div_a + b + 1e-20), 'Error here 1')
        term_1 = tf.check_numerics(
            tf.lgamma(1 + div_a + 1e-20), 'Error here 2')
        b_max = tf.check_numerics(tf.maximum(b, 1e-20), 'b is going nan')
        log_b = tf.check_numerics(tf.log(b_max), 'Error here b_log')
        term_2 = tf.check_numerics(
            tf.add(log_b, tf.lgamma(b_max)), 'Error here 3')
        numerator = tf.check_numerics(tf.add(term_1, term_2), 'Error here 4')
        full_subtract = tf.check_numerics(
            tf.subtract(numerator, denom, name='subtract'), 'Error here 5')

        return tf.exp(full_subtract, name='final_exp')


def get_pi(a, b, u_shape):
    with tf.variable_scope('train_pi'):
        u = tf.maximum(get_u(u_shape), 1e-20, name='max_u')
        max_b = tf.maximum(b, 1e-10, name='max_b')
        max_a = tf.maximum(a, 1e-10, name='max_a')
        term_a = tf.pow(max_a, -1., name='div_a')
        term_b = tf.pow(max_b, -1., name='div_b')
        log_pow_1 = tf.multiply(tf.log(u, name='log_u'),
                                term_b, name='log_pow_1')
        pow_1 = tf.exp(log_pow_1, name='term_1')
        pow_1_stable = tf.maximum(1 - pow_1, 1e-20, name='stable_pow1')
        log_pow_2 = tf.multiply(tf.log(pow_1_stable, name='log_1pow_1'),
                                term_a, name='log_pow_2')
        pow_2 = tf.exp(log_pow_2, name='pow_2')
        return tf.add(pow_2, 1e-20, name='max_pi')


def relaxed_bern(tau, probs, u_shape):
    with tf.variable_scope('relaxed_bernoulli'):
        u = tf.maximum(get_u(u_shape), 1e-20, name='max_u')

        term_1pi = tf.subtract(1., probs, name='1minus_pi')
        term_1pi_add = tf.maximum(term_1pi, 1e-20, name='1minus_pi_add')
        term_1pi = tf.log(term_1pi_add, name='log_1pi')
        term_pi_add = tf.maximum(probs, 1e-20, name='pi_add')
        term_1_pi = tf.log(term_pi_add, name='log_pi')
        term_1 = tf.subtract(term_1_pi, term_1pi, name='term_1')

        term_2u = tf.subtract(1., u, name='sub_1u')
        term_2u_max = tf.maximum(term_2u, 1e-20, name='max_pow2u')
        term_1u_log = tf.log(term_2u_max, name='term_1u_log')
        term_ulog = tf.log(u, name='u_log')
        term_2 = tf.subtract(term_ulog, term_1u_log, name='term_2_u')
        term_2_max = tf.maximum(term_2, 1e-20, name='max_term_2')

        tau_divide = tf.divide(1., tau, name='divide_tau')

        term_add = tf.add(term_1, term_2_max, name='term_add')

        unsig_z = tf.multiply(term_add, tau_divide, name='unsigmoid_z')

        z = tf.sigmoid(unsig_z, name='sigmoid_z')

        return z


def get_u(shape):
    return tf.random_uniform(shape, maxval=1)


def modularize_target(target, context: ModularContext):
    if context.mode == ModularMode.M_STEP:
        rank = target.shape.ndims
        return tf.tile(target, [context.sample_size] + [1] * (rank - 1))
    return target


def modularize_variational(
    template, optimizer, dataset_size,
    data_indices, num_batches,
    sample_size, iteration, epoch_lim, damp_length,
    alpha
):
    m = m_step(template, optimizer, dataset_size, data_indices,
               num_batches,
               sample_size, iteration, epoch_lim, damp_length, alpha)
    eval = evaluation(template, data_indices, dataset_size)
    return m, eval


def create_ema_opt():
    return tf.group(*tf.get_collection('ema'))


def get_sparsity_level():
    return tf.get_collection('sparsity')


def get_dep_input():
    return tf.get_collection('dep_input')


def get_ctrl_bias():
    return tf.get_collection('ctrl_bias')


def get_ctrl_weights():
    return tf.get_collection('ctrl_weights')
