import tensorflow as tf
from tensorflow.contrib import distributions as tfd
import numpy as np

from libmodular.modular import ModulePool, ModularContext, ModularMode, ModularLayerAttributes, VariationalLayerAttributes
from libmodular.modular import run_modules, run_masked_modules, e_step, m_step, evaluation, run_masked_modules_withloop, run_modules_withloop, run_masked_modules_withloop_and_concat
from tensorflow.python import debug as tf_debug

def create_dense_modules(inputs_or_shape, context,module_count: int, units: int = None, 
                        activation=None):
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

        def module_fnc(x, a, mask):
            """
            Takes in input and a module, multiplies input with the weights of the module
            weights are [module x input_shape x units]
            """

            new_weights = tf.einsum('mio,sm->msio', weights, tf.cast(mask, tf.float32))
            new_biases = tf.einsum('mo,sm->mso', biases, tf.cast(mask, tf.float32))
            x = tf.reshape(x, [-1, context.sample_size, x.shape[-1].value])
            out = tf.einsum('bsi,sio->bso', x, new_weights[a]) + new_biases[a]
            out = tf.reshape(out, [-1, units])
            if activation is not None:
                out = activation(out)
            return out

        return ModulePool(module_count, module_fnc, output_shape=[units], units=units)


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

        def module_fnc(x, a, mask):
            new_biases = tf.einsum('mi,m->mi', biases, mask)
            new_filter = tf.einsum('miocd,m->miocd', filter, mask)
            return tf.nn.conv2d(x, new_filter[a], strides, padding) + new_biases[a]

        return ModulePool(module_count, module_fnc, output_shape=None, units=list(shape)[-1])


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
    context: ModularContext, eps):
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

        flat_inputs = tf.stop_gradient(tf.layers.flatten(inputs))
        input_shape = flat_inputs.shape[-1].value

        shape = modules.module_count
        input_shape = flat_inputs.shape[-1].value
        u_shape = [shape]

        a = tf.get_variable(name='a', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=2.0, maxval=2.0)) + 1e-20
        b = tf.get_variable(name='b', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=0.3, maxval=0.3)) + 1e-20

        # a = tf.check_numerics(a, 'a is NaN')
        # b = tf.check_numerics(b, 'b is NaN')

        pi = get_pi(a, b, u_shape)
        
        tau = 0.001
        z = relaxed_bern(tau, pi, [shape])

        g = tf.get_default_graph()

        if context.mode == ModularMode.M_STEP:
            test_pi = pi
            selection = tf.round(z)
            final_selection = tf.tile(
                selection, [tf.shape(flat_inputs)[0]])
            final_selection = tf.reshape(
                final_selection,[tf.shape(flat_inputs)[0], shape])

        elif context.mode == ModularMode.EVALUATION:
            test_pi = get_test_pi(a, b)
            selection = tf.where(test_pi>0.5,
                                x=tf.ones_like(test_pi),
                                y=tf.zeros_like(test_pi)
                                )
            final_selection = tf.tile(
                selection, [tf.shape(inputs)[0]])
            final_selection = tf.reshape(
                final_selection,[tf.shape(inputs)[0], shape])

        pseudo_ctrl = tfd.Bernoulli(probs=pi)
        attrs = ModularLayerAttributes(selection, 
                                        None, pseudo_ctrl, 
                                        a, b, pi, None, None,
                                        None, None, None)
        context.layers.append(attrs)

        return (run_masked_modules_withloop_and_concat(inputs, 
                                    final_selection,
                                    z,
                                    shape,
                                    modules.units,
                                    modules.module_fnc, 
                                    modules.output_shape), 
                pi, selection, test_pi, test_pi)

def reinforce_mask(
    inputs, modules: ModulePool, 
    context: ModularContext, eps, tile_shape):
    with tf.variable_scope(None, 'reinforce'):

        inputs = context.begin_modular(inputs)
        flat_inputs = tf.stop_gradient(tf.layers.flatten(inputs))
        input_shape = flat_inputs.shape[-1].value

        shape = modules.module_count
        input_shape = flat_inputs.shape[-1].value
        u_shape = [shape]

        a = tf.get_variable(name='a', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=2.0, maxval=2.0)) + 1e-20
        b = tf.get_variable(name='b', 
                            dtype=tf.float32, 
                            initializer=tf.random_uniform(
                                [shape], minval=0.3, maxval=0.3)) + 1e-20

        pi = tf.distributions.Beta(a, b).sample(sample_shape=(context.sample_size))

        ctrl = tfd.Bernoulli(pi)
        z = ctrl.sample()

        if context.mode == ModularMode.M_STEP:
            test_pi = pi
            selection = tf.round(z)
            final_selection = tf.tile(
                selection, [tile_shape, 1])

        if context.mode == ModularMode.EVALUATION:
            test_pi = mean_of_beta(a, b)
            selection = mode_of_bernoulli(test_pi)  
            final_selection = tf.tile(
                selection, [tf.shape(inputs)[0]])
            final_selection = tf.reshape(
                final_selection,[tf.shape(inputs)[0], shape])

        attrs = ModularLayerAttributes(z,
                                        None, ctrl, 
                                        a, b, pi, None, None,
                                        None, None, None)
        context.layers.append(attrs)

        return (run_masked_modules_withloop_and_concat(inputs, 
                                    final_selection,
                                    z,
                                    shape,
                                    modules.units,
                                    modules.module_fnc, 
                                    modules.output_shape), 
                pi, selection, test_pi, test_pi)
        
def mean_of_beta(a, b):
    return tf.distributions.Beta(a, b).mean()

def mode_of_bernoulli(pi):
    return tfd.Bernoulli(pi).mode()


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
        u = tf.add(get_u(u_shape), 1e-20, name='max_u')
        max_b = tf.add(b, 1e-20, name='max_b')
        max_a = tf.add(a, 1e-20, name='max_a')
        term_a = tf.realdiv(1., max_a, name='div_a')
        term_b = tf.realdiv(1., max_b, name='div_b')
        log_pow_1 = tf.multiply(tf.log(u, name='log_u'), 
                                term_b, name='log_pow_1')
        pow_1 = tf.exp(log_pow_1, name='term_1')
        pow_1_stable = tf.add(pow_1, 1e-20, name='stable_pow1')
        log_pow_2 = tf.multiply(tf.log(1-pow_1, name='log_1pow_1'), 
                                term_a, name='log_pow_2')
        pow_2 = tf.exp(log_pow_2, name='pow_2')
        return tf.add(pow_2, 1e-20, name='max_pi')

def relaxed_bern(tau, probs, u_shape):
    with tf.variable_scope('relaxed_bernoulli'):
        u = tf.add(get_u(u_shape), 1e-20, name='max_u')

        term_1pi = tf.subtract(1., probs, name='1minus_pi')
        term_1pi_add = tf.add(term_1pi, 1e-20, name='1minus_pi_add')
        term_1pi = tf.log(term_1pi_add, name='log_1pi')
        term_pi_add = tf.add(probs, 1e-20, name='pi_add')
        term_1_pi = tf.log(term_pi_add, name='log_pi')
        term_1 = tf.subtract(term_1_pi, term_1pi, name='term_1')

        term_2u = tf.subtract(1., u, name='sub_1u')
        term_2u_max = tf.add(term_2u, 1e-20, name='max_pow2u')
        term_1u_log = tf.log(term_2u_max, name='term_1u_log')
        term_ulog = tf.log(u, name='u_log')
        term_2 = tf.subtract(term_ulog, term_1u_log, name='term_2_u')
        term_2_max = tf.add(term_2, 1e-20, name='max_term_2')

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

def modularize_reinforce(template, optimizer, dataset_size, 
                          data_indices, reinforce, num_batches,
                          sample_size):
    m = m_step(template, optimizer, dataset_size, 
               data_indices, reinforce, None, sample_size)
    eval = evaluation(template, data_indices, dataset_size)
    return  m, eval


def modularize_variational(template, optimizer, dataset_size, 
                          data_indices, variational, num_batches):
    m = m_step(template, optimizer, dataset_size, data_indices, 
               variational, num_batches)
    eval = evaluation(template, data_indices, dataset_size)
    return m, eval

def create_ema_opt():
    return tf.group(*tf.get_collection('ema'))
