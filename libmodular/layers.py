import tensorflow as tf
from tensorflow.contrib import distributions as tfd

from libmodular.modular import ModulePool, ModularContext, ModularMode, ModularLayerAttributes
from libmodular.modular import run_modules, run_masked_modules, e_step, m_step, evaluation


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


def modular_layer(inputs, modules: ModulePool, parallel_count: int, context: ModularContext):
    """
    Takes in modules and runs them based on the best selection
    """
    with tf.variable_scope(None, 'modular_layer'):

        #[sample_size*batch x 784]
        inputs = context.begin_modular(inputs) #At first step, tile the inputs so it can go through ALL modules

        #Takes in input and returns tensor of shape modules * parallel
        #One output per module, [sample_size*batch x modules]
        logits = tf.layers.dense(inputs, modules.module_count * parallel_count)
        logits = tf.reshape(logits, [-1, parallel_count, modules.module_count]) #[sample*batch x 1 x module]

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
        return run_modules(inputs, selection, modules.module_fnc, modules.output_shape)


def masked_layer(inputs, modules: ModulePool, context: ModularContext):
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

        logits = tf.layers.dense(inputs, modules.module_count)

        ctrl_bern = tfd.Bernoulli(logits) #Create controller with logits

        # shape = [context.dataset_size, modules.module_count]
        # initializer = tf.ones_initializer(dtype=tf.int32)
        # best_selection_persistent = tf.get_variable('best_selection', shape, tf.int32, initializer)

        initializer = tf.random_uniform_initializer(maxval=2, dtype=tf.int32)
        shape = [context.dataset_size, modules.module_count]
        best_selection_persistent = tf.get_variable('best_selection', shape, tf.int32, initializer) #Different for each layer

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
        return run_masked_modules(inputs, selection, modules.module_fnc, modules.output_shape), logits


def modularize_target(target, context: ModularContext):
    if context.mode == ModularMode.E_STEP:
        rank = target.shape.ndims
        return tf.tile(target, [context.sample_size] + [1] * (rank - 1))
    return target


def modularize(template, optimizer, dataset_size, data_indices, sample_size):
    e = e_step(template, sample_size, dataset_size, data_indices)
    m = m_step(template, optimizer, dataset_size, data_indices)
    ev = evaluation(template)
    return e, m, ev
