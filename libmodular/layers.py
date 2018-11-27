import tensorflow as tf
from tensorflow.contrib import distributions as tfd
import numpy as np

from libmodular.modular import ModulePool, ModularContext, ModularMode, ModularLayerAttributes
from libmodular.modular import run_modules, run_masked_modules, e_step, m_step, evaluation, run_masked_modules_withloop, run_masked_modules_withloop_and_concat


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

        def module_fnc(x, a, weights, biases):
            """
            Takes in input and a module, multiplies input with
            the weights of the module
            weights are [module x input_shape x units]
            """
            # Put modules in the lef tmost axis
            w = tf.transpose(weights, [1, 0, 2, 3])
            b = tf.transpose(biases, [1, 0, 2])
            out = tf.einsum('bi,bio->bo', x, w[a]) + b[a]
            if activation is not None:
                out = activation(out)
            return out

        return ModulePool(
            module_count,
            module_fnc,
            output_shape=[units], units=units,
            weight=weights, bias=biases)


def batch_norm(inputs):
    with tf.variable_scope(None, 'batch_norm'):
        return tf.layers.batch_normalization(inputs)


def create_conv_modules(shape, module_count: int, strides, padding='SAME'):
    with tf.variable_scope(None, 'conv_modules'):
        filter_shape = [module_count] + list(shape)
        filter = tf.get_variable(
            name='filter',
            shape=filter_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        biases_shape = [module_count, shape[-1]]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a, filter, bias):
            fw, fh, d = shape[0], shape[1], shape[-1]
            h, w, c = x.shape[1].value, x.shape[2].value, x.shape[-1].value
            b = tf.shape(x)[0]
            new_filter = tf.transpose(filter,
                                      [1, 0, 2, 3, 4, 5])
            new_biases = tf.transpose(bias,
                                      [1, 0, 2])
            new_biases = new_biases[a]
            new_filter = new_filter[a]
            # [fh,fw,b,c,d]
            filter_transpose = tf.transpose(new_filter, [1, 2, 0, 3, 4])
            # [fh,fw,b*c,d]
            filter_reshape = tf.reshape(filter_transpose,
                                        [fw, fh, b * c, d])
            # [h,w,b,c]
            inputs_transpose = tf.transpose(x, [1, 2, 0, 3])
            # [1,h,w,b*c]
            inputs_reshape = tf.reshape(inputs_transpose,
                                        [1, h, w, b * c])

            out = tf.nn.depthwise_conv2d(
                inputs_reshape,
                filter=filter_reshape,
                strides=[1, 1, 1, 1],
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

        return ModulePool(
            module_count, module_fnc, output_shape=None,
            units=list(shape)[-1], weight=filter, bias=biases)


def conv_layer(x, shape, strides, padding='SAME'):
    with tf.variable_scope(None, 'simple_conv_layer'):
        filter_shape = list(shape)
        filter = tf.get_variable(
            'filter', filter_shape, initializer=tf.contrib.layers.xavier_initializer())
        biases_shape = [shape[-1]]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())
        hidden = tf.nn.conv2d(x, filter, strides, padding) + biases
        pooled = tf.nn.max_pool(
            hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.relu(pooled)


def modular_layer(inputs, modules: ModulePool, parallel_count: int, context: ModularContext):
    """
    Takes in modules and runs them based on the best selection
    """
    with tf.variable_scope(None, 'modular_layer'):

        # [sample_size*batch x 784]
        # At first step, tile the inputs so it can go through ALL modules
        inputs = context.begin_modular(inputs)

        flat_inputs = tf.layers.flatten(inputs)
        # Takes in input and returns tensor of shape modules * parallel
        # One output per module, [sample_size*batch x modules]
        logits = tf.layers.dense(
            flat_inputs, modules.module_count * parallel_count)
        # [sample*batch x parallel x module]
        logits = tf.reshape(logits, [-1, parallel_count, modules.module_count])

        # For each module and batch, have one logit, so [Batch x modules]
        ctrl = tfd.Categorical(logits)  # Create controller with logits

        initializer = tf.random_uniform_initializer(
            maxval=modules.module_count, dtype=tf.int32)
        shape = [context.dataset_size, parallel_count]
        best_selection_persistent = tf.get_variable(
            'best_selection', shape, tf.int32, initializer)  # Different for each layer

        if context.mode == ModularMode.E_STEP:
            # Use gather to get the selection of the batch indices of the data [1...32] then [32...64]
            # [1 x Batch x parallel]
            best_selection = tf.gather(best_selection_persistent, context.data_indices)[
                tf.newaxis]  # [1 x B x parallel]
            # One sample for each tiled batch [Sample*batch x parallel]
            samples = ctrl.sample()
            # [Sample x Batch x parallel]
            sampled_selection = tf.reshape(
                samples, [context.sample_size, -1, parallel_count])
            # Need selection to be sample size in 1st Dim
            selection = tf.concat(
                [best_selection, sampled_selection[1:]], axis=0)
            # Back to [Sample*batch x parallel]
            selection = tf.reshape(selection, [-1, parallel_count])
        elif context.mode == ModularMode.M_STEP:
            selection = tf.gather(
                best_selection_persistent, context.data_indices)
        elif context.mode == ModularMode.EVALUATION:
            selection = ctrl.mode()
        else:
            raise ValueError('Invalid modular mode')

        attrs = ModularLayerAttributes(
            selection, best_selection_persistent, ctrl)
        context.layers.append(attrs)

        return run_modules(inputs, selection, modules.module_fnc, modules.output_shape), logits, best_selection_persistent


def masked_layer(inputs, modules: ModulePool, context: ModularContext, initializer):
    """
    Function that takes in input and modules and return the selection based on a Bernoulli mask
    Args:
        inputs; tensor to be fed to controller
        modules; namedtuple containing the module functions
        context; ModularContext class
    Returns:
        output of run_modules, which is the logits of the given layer
    """
    with tf.variable_scope(None, 'masked_layer'):

        inputs = context.begin_modular(inputs)

        flat_inputs = tf.layers.flatten(inputs)

        logits = tf.layers.dense(flat_inputs, modules.module_count)
        logits = tf.sigmoid(logits)

        ctrl_bern = tfd.Bernoulli(logits)  # Create controller with logits

        shape = [context.dataset_size, modules.module_count]
        best_selection_persistent = tf.get_variable(
            'best_selection', shape=shape, dtype=tf.int32, initializer=initializer)  # Different for each layer

        if context.mode == ModularMode.E_STEP:
            best_selection = tf.gather(
                best_selection_persistent, context.data_indices)[tf.newaxis]
            samples = ctrl_bern.sample()
            sampled_selection = tf.reshape(
                samples, [context.sample_size, -1, modules.module_count])
            selection = tf.concat(
                [best_selection, sampled_selection[1:]], axis=0)
            selection = tf.reshape(selection, [-1, modules.module_count])
            selection = tf.cast(selection, tf.int32)
        elif context.mode == ModularMode.M_STEP:
            selection = tf.gather(
                best_selection_persistent, context.data_indices)
        elif context.mode == ModularMode.EVALUATION:
            selection = tf.cast(tf.where(logits > 0.5,
                                         x=tf.ones_like(logits),
                                         y=tf.zeros_like(logits)), tf.int32)
        else:
            raise ValueError('Invalid modular mode')

        attrs = ModularLayerAttributes(
            selection, best_selection_persistent, ctrl_bern)
        context.layers.append(attrs)

        return (run_masked_modules_withloop(
            inputs,
            selection,
            selection,
            modules.module_count,
            modules.units,
            modules.module_fnc,
            modules.output_shape,
            modules.weight,
            modules.bias),
            logits, best_selection_persistent)


def modularize_target(target, context: ModularContext):
    if context.mode == ModularMode.E_STEP:
        rank = target.shape.ndims
        return tf.tile(target, [context.sample_size] + [1] * (rank - 1))
    return target


def modularize(template, optimizer, dataset_size, data_indices, sample_size):
    e = e_step(template, sample_size, dataset_size, data_indices)
    m = m_step(template, optimizer, dataset_size, data_indices)
    ev = evaluation(template, data_indices)
    return e, m, ev
