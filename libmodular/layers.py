import tensorflow as tf
from tensorflow.contrib import distributions as tfd
import numpy as np

from libmodular.modular import ModulePool, ModularContext, ModularMode, ModularLayerAttributes, VariationalLayerAttributes
from libmodular.modular import run_modules, run_masked_modules, e_step, m_step, evaluation, run_masked_modules_withloop, run_modules_withloop


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
        weights = tf.get_variable('weights', weights_shape)
        biases_shape = [module_count, units]
        biases = tf.get_variable('biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a):
            """
            Takes in input and a module, multiplies input with the weights of the module
            weights are [module x input_shape x units]
            """
            out = tf.matmul(x, weights[a]) + biases[a]
            if activation is not None:
                out = activation(out)
            return out

        return ModulePool(module_count, module_fnc, output_shape=[units])


def create_conv_modules(shape, module_count: int, strides, padding='SAME'):
    with tf.variable_scope(None, 'conv_modules'):
        filter_shape = [module_count] + list(shape)
        filter = tf.get_variable('filter', filter_shape)
        biases_shape = [module_count, shape[-1]]
        biases = tf.get_variable('biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a):

            return tf.nn.conv2d(x, filter[a], strides, padding) + biases[a]

        return ModulePool(module_count, module_fnc, output_shape=None)


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

        return run_modules_withloop(inputs, selection, modules.module_fnc, modules.output_shape), logits, best_selection_persistent


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

def variational_mask(inputs, modules: ModulePool, context: ModularContext, eps, rho):
    """
    Constructs a Bernoulli masked layer that outputs sparse binary masks dependent on the input
    Based on the Adaptive network Sparsification paper
    Args:
        inputs; Batch X dimension
        modules; ModulePool named tuple
        contect; context class
        eps; threshold for dependent pi
    """
    with tf.variable_scope(None, 'variational_mask'):

        in_shape = modules.module_count

        a = tf.get_variable(name='a', dtype=tf.float32, initializer=tf.random_uniform([in_shape], minval=18, maxval=20.5))
        b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.random_uniform([in_shape], minval=18, maxval=20.5))

        u = get_u(in_shape)

        pi = tf.pow((1 - tf.pow(u, tf.divide(1,b))), tf.divide(1,a))

        eta = tf.get_variable(name='eta', shape=[in_shape], dtype=tf.float32)
        khi = tf.get_variable(name='khi', shape=[in_shape], dtype=tf.float32)
        beta = tfd.MultivariateNormalDiag(eta, khi)

        gamma = tf.get_variable(name='gamma', shape=[in_shape], dtype=tf.float32)

        def dependent_pi(inputs, pi, gamma, beta, eps):
            #Lower input dimenstions
            flat_inputs = tf.stop_gradient(tf.layers.flatten(inputs))
            inputs = tf.layers.dense(flat_inputs, modules.module_count)
            mean, var = tf.nn.moments(inputs, axes=0, keep_dims=True)
            standardised = tf.stop_gradient(tf.div(tf.subtract(inputs, mean), tf.sqrt(var)))
            new_input = tf.multiply(gamma, standardised) + beta
            max_input = tf.maximum(eps, new_input)
            min_input = tf.minimum(1-eps, max_input)
            new_pi = tf.multiply(pi, min_input)
            return tf.layers.dense(new_pi, modules.module_count)

        def new_dependent_pi(inputs, pi):
            """
            Cant use this as pi has to be between 0 and 1!!!
            """
            flat_inputs = tf.stop_gradient(tf.layers.flatten(inputs))
            lowered_inputs = tf.layers.dense(flat_inputs, modules.module_count)
            return tf.sigmoid(tf.multiply(pi, lowered_inputs))

        #Make selection
        final_pi = tf.make_template('dependent', new_dependent_pi)
        new_pi = final_pi(inputs, pi)

        tau = 0.1
        term_1 = tf.log(tf.divide(new_pi, 1-new_pi))
        term_2 = tf.log(tf.divide(u, 1-u))
        z = tf.sigmoid(tf.multiply(tf.divide(1, tau), term_1 + term_2))

        if context.mode == ModularMode.M_STEP:
            selection = tf.cast(z, tf.int32)
        elif context.mode == ModularMode.EVALUATION:
            test_pi = get_test_selection(a, b)
            selection = tf.cast(final_pi(inputs, get_test_selection(a,b)), dtype=tf.int32)

        #Need prior on Beta for the KL divergence
        beta_prior = tfd.MultivariateNormalDiag(tf.zeros([in_shape]), 
                                                tf.multiply(rho, tf.ones([in_shape]))
                                                )
        attrs = VariationalLayerAttributes(selection, z, a, b, beta, beta_prior)
        context.var_layers.append(attrs)

        return run_masked_modules_withloop(inputs, selection, modules.module_fnc, modules.output_shape), new_pi, attrs


def new_controller(inputs, modules: ModulePool, context: ModularContext, initializer):

    with tf.variable_scope(None, 'New_controller'):

        shape = modules.module_count
        a = tf.get_variable(name='a', dtype=tf.float32, initializer=tf.random_uniform([shape], 
                                                                                      minval=20, maxval=30))
        b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.random_uniform([shape], 
                                                                                      minval=20, maxval=30))
        u = get_u(shape)
        pi = tf.pow((1 - tf.pow(u, tf.divide(1,b + 1e-20))), tf.divide(1,a + 1e-20))
        
        input_shape = inputs.shape[-1].value
        eta = tf.get_variable(name='eta', shape=[input_shape], dtype=tf.float32)
        khi = tf.get_variable(name='khi', shape=[input_shape], dtype=tf.float32)
        beta = tfd.MultivariateNormalDiag(eta, khi)

        gamma = tf.get_variable(name='gamma', shape=[input_shape], dtype=tf.float32)


        def dependent_pi(inputs, pi, gamma, beta, eps):
            stop_input = tf.stop_gradient(inputs)
            mean, var = tf.nn.moments(stop_input, axes=0, keep_dims=True)
            standardised = tf.div(tf.subtract(stop_input, mean), tf.sqrt(var) + 1e-20)
            new_input =  tf. multiply(gamma, standardised) + beta
            max_input = tf.maximum(eps, new_input)
            min_input = tf.minimum(1-eps, max_input)
            reshape = tf.reshape(min_input, 
                                 [-1, shape, tf.cast(tf.divide(tf.shape(inputs)[-1], shape), tf.int32)])
            dep = tf.reduce_sum(reshape, axis=2)

            def give_input_dependent(standardised):
                zero = tf.constant(0.1, shape=[],
                                   dtype=tf.float32)
                fill_zero = tf.where(standardised>zero,
                               x=standardised,
                               y=tf.zeros_like(standardised))
                fill_one = tf.where(standardised>zero,
                                   x=tf.ones_like(standardised),
                                   y=standardised)
                return fill_one

            final_dep = give_input_dependent(dep)
            return tf.multiply(pi, final_dep), final_dep

        rho = 3.17
        beta_prior = tfd.MultivariateNormalDiag(tf.zeros([input_shape]), 
                                                tf.multiply(rho, tf.ones([input_shape]))
                                               )

        #Get controller logits
        # inputs = tf.stop_gradient(context.begin_modular(inputs))
        # flat_inputs = tf.layers.flatten(inputs)
        # lowered_inputs = tf.layers.dense(flat_inputs, modules.module_count)
        # logits = tf.multiply(pi, lowered_inputs)

        logits = dependent_pi(inputs, pi, gamma, beta.sample(), eps=0.0001)

        ctrl_bern = tfd.Bernoulli(logits=logits)

        shape = [context.dataset_size, modules.module_count]
        best_selection_persistent = tf.get_variable('best_selection', shape=shape, dtype=tf.int32, initializer=initializer)

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

        attrs = ModularLayerAttributes(selection, best_selection_persistent, ctrl_bern, a, b, beta, beta_prior)
        context.layers.append(attrs)

        return run_masked_modules_withloop(inputs, selection, modules.module_fnc, modules.output_shape), tf.sigmoid(logits), best_selection_persistent


def get_test_selection(a, b):
    denom = tf.exp(tf.lgamma(1 + tf.pow(a,-1) + b))
    term_1 = tf.exp(tf.lgamma(1 + tf.pow(a,-1)))
    term_2 = tf.multiply(b, tf.exp(tf.lgamma(b)))
    numerator = tf.multiply(term_1, term_2)
    return tf.divide(numerator, denom)

def get_u(shape):
    return tf.random_uniform([shape], maxval=1)

def modularize_target(target, context: ModularContext):
    if context.mode == ModularMode.E_STEP:
        rank = target.shape.ndims
        return tf.tile(target, [context.sample_size] + [1] * (rank - 1))
    return target


def modularize(template, optimizer, dataset_size, data_indices, sample_size, variational):
    e = e_step(template, sample_size, dataset_size, data_indices)
    m = m_step(template, optimizer, dataset_size, data_indices, variational)
    ev = evaluation(template, data_indices)
    return e, m, ev

def modularize_variational(template, optimizer, dataset_size, data_indices, variational):
    m = m_step(template, optimizer, dataset_size, data_indices, variational)
    ev = evaluation(template, data_indices)
    return m, ev
