import tensorflow as tf
from tensorflow.contrib import distributions as tfd
import numpy as np

from libmodular.modular import ModulePool, ModularContext, ModularMode, ModularLayerAttributes, VariationalLayerAttributes
from libmodular.modular import run_modules, run_masked_modules, e_step, m_step, evaluation, run_masked_modules_withloop, run_modules_withloop, run_masked_modules_withloop_and_concat, run_non_modular
from tensorflow.python import debug as tf_debug


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
        else:
            return args, kwargs
    wrapper.has_run = False
    return wrapper


def create_dense_modules(inputs_or_shape, module_count: int, units: int = None, activation=None):
    """
    Takes in input, module count, units, and activation and returns a named tuple with a function
    that returns the multiplcation of a sepcific module with the input
    """
    with tf.variable_scope(None, 'dense_modules'):
        if hasattr(inputs_or_shape, 'shape') and units is not None: #Checks if input has attribute shape and takes last
            weights_shape = [module_count, inputs_or_shape.shape[-1].value, units] #First dimension is the weights of a specific module
        else:
            weights_shape = [module_count] + inputs_or_shape
        weights = tf.get_variable(
            'weights', weights_shape, initializer=tf.contrib.layers.xavier_initializer())
        biases_shape = [module_count, units]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a, mask, w, b):
            """
            Takes in input and a module, multiplies input with the weights of the module
            weights are [module x input_shape x units]
            """
            #Put modules in the lef tmost axis
            w = tf.transpose(w, [1,0,2,3])
            b = tf.transpose(b, [1,0,2])
            out = tf.einsum('bi,bio->bo', x, w[a]) + b[a]
            if activation is not None:
                out = activation(out)
            return out

        return ModulePool(module_count, module_fnc, output_shape=[units], units=units,
            weight=weights, bias=biases)


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


def create_conv_modules(shape, module_count: int, strides, padding='SAME'):
    with tf.variable_scope(None, 'conv_modules'):
        filter_shape = [module_count] + list(shape)
        filter = tf.get_variable(
            'filter', filter_shape, initializer=tf.contrib.layers.xavier_initializer())
        biases_shape = [module_count, shape[-1]]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a, mask, weight, bias):
            fw,fh,d = shape[0], shape[1], shape[-1]
            h,w,c = x.shape[1].value, x.shape[2].value, x.shape[-1].value
            b = tf.shape(x)[0]
            new_filter = tf.transpose(weight,
                                    [1,0,2,3,4,5])
            new_biases = tf.transpose(bias,
                                    [1,0,2])
            new_biases = new_biases[a]
            new_filter = new_filter[a]
            #[fh,fw,b,c,d]
            filter_transpose = tf.transpose(new_filter, [1,2,0,3,4])
            #[fh,fw,b*c,d]
            filter_reshape = tf.reshape(filter_transpose, 
                                        [fw, fh, b*c, d])
            #[h,w,b,c]
            inputs_transpose = tf.transpose(x, [1,2,0,3])
            #[1,h,w,b*c]
            inputs_reshape = tf.reshape(inputs_transpose, 
                                        [1, h, w, b*c])

            out = tf.nn.depthwise_conv2d(
                      inputs_reshape,
                      filter=filter_reshape,
                      strides=[1, 1, 1, 1],
                      padding=padding)
            if padding == "SAME":
                out = tf.reshape(out, [h, w, -1, c, d])
            if padding == "VALID":
                out = tf.reshape(out, [h-fh+1, w-fw+1, -1, c, d])
            out = tf.transpose(out, [2, 0, 1, 3, 4])
            out = tf.reduce_sum(out, axis=3) 
            out = tf.transpose(out, [1,2,0,3]) + new_biases
            out = tf.transpose(out, [2,0,1,3])

            return out

        def module_fnc_original(x, a, mask, weight, bias):
            return tf.nn.conv2d(x, weight[a], strides, padding) + bias[a]

        def module_fnc_non_modular(x, a, mask, weight, bias):
            new_filter = tf.transpose(weight,
                                    [1,0,2,3,4,5])
            new_biases = tf.transpose(bias,
                                    [1,0,2])
            new_biases = new_biases[a]
            new_filter = new_filter[a]
            return tf.nn.conv2d(x, new_filter, strides, padding) + new_biases[a]

        return ModulePool(module_count, module_fnc, output_shape=None, 
                        units=list(shape)[-1], weight=filter, bias=biases)


def batch_norm(inputs):
    with tf.variable_scope(None, 'batch_norm'):
        return tf.layers.batch_normalization(inputs)


def modular_layer(inputs, modules: ModulePool, parallel_count: int, context: ModularContext):
    """
    Takes in modules and runs them based on the best selection
    """
    with tf.variable_scope(None, 'modular_layer'):

        #[sample_size*batch x 784]
        inputs = context.begin_modular(inputs) #At first step, tile the inputs so it can go through ALL modules

        flat_inputs = tf.layers.flatten(inputs)
        #Takes in input and returns tensor of shape modules * parallel
        #One output per module, [sample_size*batch x modules]
        logits = tf.layers.dense(flat_inputs, modules.module_count * parallel_count)
        logits = tf.reshape(logits, [-1, parallel_count, modules.module_count]) #[sample*batch x parallel x module]

        #For each module and batch, have one logit, so [Batch x modules]
        ctrl = tfd.Categorical(logits) #Create controller with logits

        initializer = tf.random_uniform_initializer(maxval=modules.module_count, dtype=tf.int32)
        shape = [context.dataset_size, parallel_count]
        best_selection_persistent = tf.get_variable('best_selection', shape, tf.int32, initializer) #Different for each layer

        if context.mode == ModularMode.E_STEP:
            #Use gather to get the selection of the batch indices of the data [1...32] then [32...64]
            #[1 x Batch x parallel]
            best_selection = tf.gather(best_selection_persistent, context.data_indices)[tf.newaxis] #[1 x B x parallel]
            samples = ctrl.sample() #One sample for each tiled batch [Sample*batch x parallel]
            sampled_selection = tf.reshape(samples, [context.sample_size, -1, parallel_count]) #[Sample x Batch x parallel]
            selection = tf.concat([best_selection, sampled_selection[1:]], axis=0) #Need selection to be sample size in 1st Dim
            selection = tf.reshape(selection, [-1, parallel_count]) #Back to [Sample*batch x parallel]
        elif context.mode == ModularMode.M_STEP:
            selection = tf.gather(best_selection_persistent, context.data_indices)
        elif context.mode == ModularMode.EVALUATION:
            selection = ctrl.mode()
        else:
            raise ValueError('Invalid modular mode')

        attrs = ModularLayerAttributes(selection, best_selection_persistent, ctrl)
        context.layers.append(attrs)

        return run_masked_modules_withloop_and_concat(inputs, selection, modules.module_fnc, modules.output_shape), logits, best_selection_persistent


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
        # probs = tf.sigmoid(logits)
        # greater = tf.greater(probs, 0.5)
        # gate = tf.cast(greater, tf.int32)


        ctrl_bern = tfd.Bernoulli(logits=logits) #Create controller with logits

        #Initialisation of variables to 1
        # shape = [context.dataset_size, modules.module_count]
        # initializer = tf.random_uniform_initializer(maxval=1, dtype=tf.int32)
        
        shape = [context.dataset_size, modules.module_count]
        # initializer = tf.constant_initializer()
        best_selection_persistent = tf.get_variable('best_selection', shape=shape, dtype=tf.int32, initializer=initializer) #Different for each layer

        if context.mode == ModularMode.E_STEP:
            best_selection = tf.gather(best_selection_persistent, context.data_indices)[tf.newaxis]
            samples = ctrl_bern.sample()
            sampled_selection = tf.reshape(samples, [context.sample_size, -1, modules.module_count]) 
            selection = tf.concat([best_selection, sampled_selection[1:]], axis=0)
            selection = tf.reshape(selection, [-1, modules.module_count])
        elif context.mode == ModularMode.M_STEP:
            selection = tf.gather(best_selection_persistent, context.data_indices)
        elif context.mode == ModularMode.EVALUATION:
            selection = ctrl_bern.mode()
        else:
            raise ValueError('Invalid modular mode')

        attrs = ModularLayerAttributes(selection, best_selection_persistent, ctrl_bern)
        context.layers.append(attrs)
        return run_masked_modules_withloop(inputs, selection, modules.module_fnc, modules.output_shape), logits, best_selection_persistent

def variational_mask(
    inputs, modules: ModulePool, 
    context: ModularContext, eps, tile_shape):
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
    with tf.variable_scope(None, 'variational_mask'):

        inputs = context.begin_modular(inputs)
        flat_inputs = tf.stop_gradient(tf.layers.flatten(inputs))
        input_shape = flat_inputs.shape[-1].value

        shape = modules.module_count
        input_shape = flat_inputs.shape[-1].value
        u_shape = [context.sample_size, shape]

        a = tf.get_variable(name='a', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=3.5, maxval=3.5)) + 1e-20
        b = tf.get_variable(name='b', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=0.3, maxval=0.3)) + 1e-20

        pi = get_pi(a, b, u_shape)
        pi_batch = tf.expand_dims(pi, 1)
        pi_batch = tf.tile(
                pi_batch, [1, tile_shape, 1])
        pi_batch = tf.reshape(
                pi_batch, 
                [tile_shape*context.sample_size, shape])

        tau = 0.01
        z = relaxed_bern(tau, pi_batch, [tile_shape*context.sample_size, shape])
        # z = tf.expand_dims(z, 1)
        # z = tf.tile(
        #         z, [1, tile_shape, 1])
        # z = tf.reshape(
        #         z, 
        #         [tile_shape*context.sample_size, shape])

        if context.mode == ModularMode.M_STEP:
            test_pi = pi
            selection = tf.round(z)
            final_selection = selection

        elif context.mode == ModularMode.EVALUATION:
            test_pi = get_test_pi(a, b)
            selection = tf.where(test_pi>0.5,
                                x=tf.ones_like(test_pi),
                                y=tf.zeros_like(test_pi)
                                )
            final_selection = tf.tile(
                selection, [tf.shape(flat_inputs)[0]])
            final_selection = tf.reshape(
                final_selection,[tf.shape(flat_inputs)[0], shape])

            tf.add_to_collection(
                name='sparsity',
                value=final_selection)

        pseudo_ctrl = tfd.Bernoulli(probs=pi)
        attrs = ModularLayerAttributes(selection, 
                                        None, pseudo_ctrl, 
                                        a, b, pi, None, None,
                                        None, None, None)
        context.layers.append(attrs)

        if inputs.shape.ndims > 3:
            new_biases = tf.einsum('mo,bm->bmo', modules.bias, z)
            new_weights = tf.einsum('miocd,bm->bmiocd', modules.weight, z)
        else:
            new_weights = tf.einsum('mio,bm->bmio', modules.weight, tf.cast(z, tf.float32))
            new_biases = tf.einsum('mo,bm->bmo', modules.bias, tf.cast(z, tf.float32))

        return (run_masked_modules_withloop_and_concat(inputs, 
                                    final_selection,
                                    z,
                                    shape,
                                    modules.units,
                                    modules.module_fnc, 
                                    modules.output_shape,
                                    new_weights,
                                    new_biases), 
                pi, selection, test_pi, test_pi)

        # return (run_non_modular(inputs,
        #                     final_selection,
        #                     z,
        #                     shape,
        #                     modules.units,
        #                     modules.module_fnc, 
        #                     modules.output_shape), 
        #                 pi, selection, test_pi, test_pi)


def dep_variational_mask(
    inputs, modules: ModulePool, 
    context: ModularContext, eps, tile_shape):
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
        flat_inputs = tf.stop_gradient(tf.layers.flatten(inputs))
        input_shape = flat_inputs.shape[-1].value

        shape = modules.module_count
        input_shape = flat_inputs.shape[-1].value
        u_shape = [context.sample_size, shape]

        a = tf.maximum(tf.get_variable(name='a', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=3.5, maxval=3.5)), 1e-20)
        b = tf.maximum(tf.get_variable(name='b', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=0.3, maxval=0.3)), 1e-20)

        pi = get_pi(a, b, u_shape)
        pi_batch = tf.expand_dims(pi, 1)
        pi_batch = tf.tile(
                pi_batch, [1, tile_shape, 1])
        pi_batch = tf.reshape(
                pi_batch, 
                [tile_shape*context.sample_size, shape])

        initializer = tf.contrib.layers.xavier_initializer()
        def dependent_pi(inputs, pi):
            with tf.variable_scope('dep_pi', reuse=tf.AUTO_REUSE):
                dep_input = tf.layers.dense(inputs,
                                            shape,
                                            activation=tf.sigmoid,
                                            kernel_initializer=initializer)
                return tf.multiply(dep_input, pi), dep_input

        dep_pi = tf.make_template('dependent_pi', dependent_pi)

        new_pi, dep_input = dep_pi(flat_inputs, pi_batch)
        tau = 0.01
        z = relaxed_bern(tau, new_pi, [tile_shape*context.sample_size, shape])

        if context.mode == ModularMode.M_STEP:
            test_pi = new_pi
            selection = tf.round(z)
            final_selection = selection

        elif context.mode == ModularMode.EVALUATION:
            test_pi = get_test_pi(a, b)
            new_pi, dep_input = dep_pi(flat_inputs, test_pi)
            selection = tf.where(new_pi>0.5,
                                x=tf.ones_like(new_pi),
                                y=tf.zeros_like(new_pi)
                                )
            final_selection = selection
            # final_selection = tf.tile(
            #     selection, [tf.shape(flat_inputs)[0]])
            # final_selection = tf.reshape(
            #     final_selection,[tf.shape(flat_inputs)[0], shape])

            tf.add_to_collection(
                name='sparsity',
                value=final_selection)

        pseudo_ctrl = tfd.Bernoulli(probs=pi)
        attrs = ModularLayerAttributes(selection, 
                                        None, pseudo_ctrl, 
                                        a, b, pi, None, None,
                                        None, None, None)
        context.layers.append(attrs)

        if inputs.shape.ndims > 3:
            new_biases = tf.einsum('mo,bm->bmo', modules.bias, z)
            new_weights = tf.einsum('miocd,bm->bmiocd', modules.weight, z)
        else:
            new_weights = tf.einsum('mio,bm->bmio', modules.weight, tf.cast(z, tf.float32))
            new_biases = tf.einsum('mo,bm->bmo', modules.bias, tf.cast(z, tf.float32))


        return (run_masked_modules_withloop_and_concat(inputs, 
                                    final_selection,
                                    z,
                                    shape,
                                    modules.units,
                                    modules.module_fnc, 
                                    modules.output_shape,
                                    new_weights,
                                    new_biases), 
                new_pi, selection, test_pi, dep_input)

        # return (run_non_modular(inputs,
        #                     final_selection,
        #                     z,
        #                     shape,
        #                     modules.units,
        #                     modules.module_fnc, 
        #                     modules.output_shape), 
        #                 pi, selection, test_pi, test_pi)


def beta_bernoulli(
    inputs, modules: ModulePool, 
    context: ModularContext, eps, tile_shape):

    with tf.variable_scope(None, 'beta_bernoulli'):

        inputs = context.begin_modular(inputs)
        flat_inputs = tf.stop_gradient(tf.layers.flatten(inputs))
        input_shape = flat_inputs.shape[-1].value

        shape = modules.module_count
        input_shape = flat_inputs.shape[-1].value

        std = 0.001
        initializer_a = tf.truncated_normal_initializer(mean=20.1, stddev=std)
        initializer_b = tf.truncated_normal_initializer(mean=10.1, stddev=std)

        a = tf.log(tf.maximum(tf.layers.dense(
                flat_inputs, modules.module_count,
                activation=tf.nn.relu, kernel_initializer=initializer_a, name='var_a'), 1e-10))
        b = tf.log(tf.maximum(tf.layers.dense(
                flat_inputs, modules.module_count,
                activation=tf.nn.relu, kernel_initializer=initializer_b, name='var_b'),1e-10))

        # a = tf.check_numerics(a, 'a variable')
        # b = tf.check_numerics(b, 'b variable')


        u_shape = [tf.shape(a)[0], tf.shape(a)[1]]

        pi = get_pi(a, b, u_shape)

        tau = 0.1
        z = relaxed_bern(tau, pi, [tf.shape(pi)[0], tf.shape(pi)[1]])

        # z = tf.tile(
        #     z,
        #     [tile_shape, 1])

        if context.mode == ModularMode.M_STEP:
            test_pi = pi
            selection = tf.round(z)
            final_selection = selection
            # final_selection = tf.tile(
            #                     selection,
            #                     [tile_shape])
            # final_selection = tf.reshape(
            #                     final_selection,
            #                     [tile_shape, shape])


        elif context.mode == ModularMode.EVALUATION:
            test_pi = get_test_pi(a, b)
            selection = tf.where(test_pi>0.5,
                                x=tf.ones_like(test_pi),
                                y=tf.zeros_like(test_pi)
                                )
            final_selection = selection
            # final_selection = tf.tile(
            #     selection, [tf.shape(flat_inputs)[0]])
            # final_selection = tf.reshape(
            #     final_selection,[tf.shape(flat_inputs)[0], shape])

        pseudo_ctrl = tfd.Bernoulli(probs=pi)
        attrs = ModularLayerAttributes(selection, 
                                        None, pseudo_ctrl, 
                                        a, b, pi, None, None,
                                        None, None, None)
        context.layers.append(attrs)

        if inputs.shape.ndims > 3:
            new_biases = tf.einsum('mo,bm->bmo', modules.bias, z)
            new_weights = tf.einsum('miocd,bm->bmiocd', modules.weight, z)
        else:
            new_weights = tf.einsum('mio,bm->bmio', modules.weight, tf.cast(z, tf.float32))
            new_biases = tf.einsum('mo,bm->bmo', modules.bias, tf.cast(z, tf.float32))

        return (run_masked_modules_withloop_and_concat(inputs, 
                                    final_selection,
                                    z,
                                    shape,
                                    modules.units,
                                    modules.module_fnc, 
                                    modules.output_shape,
                                    new_weights,
                                    new_biases), 
                pi, selection, test_pi, test_pi)




def get_test_pi(a, b):
    with tf.variable_scope('test_pi'):
        max_a = tf.check_numerics(tf.maximum(a, 1e-20), 'a is going NaN')
        div_a = tf.check_numerics(tf.realdiv(1., max_a), 'div_a')
        denom = tf.check_numerics(tf.lgamma(1 + div_a + b + 1e-20), 'Error here 1')
        term_1 = tf.check_numerics(tf.lgamma(1 + div_a + 1e-20), 'Error here 2')
        b_max = tf.check_numerics(tf.maximum(b, 1e-20), 'b is going nan')
        log_b = tf.check_numerics(tf.log(b_max), 'Error here b_log')
        term_2 = tf.check_numerics(tf.add(log_b, tf.lgamma(b_max)), 'Error here 3')
        numerator = tf.check_numerics(tf.add(term_1, term_2), 'Error here 4')
        full_subtract = tf.check_numerics(tf.subtract(numerator, denom, name='subtract'), 'Error here 5')

        return tf.exp(full_subtract, name='final_exp')

def get_pi(a, b, u_shape):
    with tf.variable_scope('train_pi'):
        u = tf.maximum(get_u(u_shape), 1e-20, name='max_u')
        max_b = tf.maximum(b, 1e-10, name='max_b')
        max_a = tf.maximum(a, 1e-10, name='max_a')
        term_a = tf.pow(max_a,-1.,  name='div_a')
        term_b = tf.pow(max_b,-1., name='div_b')
        log_pow_1 = tf.multiply(tf.log(u, name='log_u'), 
                                term_b, name='log_pow_1')
        pow_1 = tf.exp(log_pow_1, name='term_1')
        pow_1_stable = tf.maximum(1-pow_1, 1e-20, name='stable_pow1')
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
        return tf.tile(target, [context.sample_size] + [1] * (rank-1))
    return target


def modularize(template, optimizer, dataset_size, data_indices, 
               sample_size, variational, num_batches=None):
    e = e_step(template, sample_size, dataset_size, data_indices)
    m = m_step(template, optimizer, dataset_size,
               data_indices, variational, None)
    return e, m

def modularize_variational(template, optimizer, dataset_size, 
                          data_indices, variational, num_batches, beta,
                         sample_size, iteration, epoch_lim):
    m = m_step(template, optimizer, dataset_size, data_indices, 
               variational, num_batches, beta, sample_size, iteration, epoch_lim)
    eval = evaluation(template, data_indices, dataset_size)
    return m, eval

def create_ema_opt():
    return tf.group(*tf.get_collection('ema'))

def get_sparsity_level():
    return tf.get_collection('sparsity')



