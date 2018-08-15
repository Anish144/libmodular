from collections import namedtuple
from enum import Enum
from typing import List
import libmodular.tensor_utils as tensor_utils
from tensorflow.contrib import distributions as tfd

import tensorflow as tf

M_STEP_SUMMARIES = 'M_STEP_SUMMARIES'
ModularMode = Enum('ModularMode', 'E_STEP M_STEP EVALUATION')
ModularLayerAttributes = namedtuple('ModularLayerAttributes', 
                                    ['selection', 'best_selection', 
                                    'controller', 'a', 'b', 'probs', 'beta', 
                                    'beta_prior', 'eta', 'khi', 'gamma']
                                    )
VariationalLayerAttributes = namedtuple('ModularLayerAttributes', 
                                    ['selection', 'controller', 'a', 'b', 'beta', 'beta_prior']
                                    )
ModulePool = namedtuple('ModulePool', ['module_count', 'module_fnc', 'output_shape',
                                        'units', 'weight', 'bias'])


class ModularContext:

    def __init__(self, mode: ModularMode, data_indices=None, dataset_size: int = None, 
                 sample_size: int = 1, variational=False):
        self.mode = mode
        self.data_indices = data_indices
        self.dataset_size = dataset_size
        self.sample_size = sample_size
        self.e_step_samples = False
        self.layers: List[ModularLayerAttributes] = [] #Save the module layer attributes in "Modular_Layer"
        self.var_layers: List[VariationalLayerAttributes] = []
        #Switch for m step to be variational or non variational
        self.variational = variational

    def begin_modular(self, inputs):
        if self.mode == ModularMode.M_STEP and not self.e_step_samples:
            self.e_step_samples = True
            rank = inputs.shape.ndims
            return tf.tile(inputs, [self.sample_size] +[1] * (rank - 1))
        return inputs

    def selection_entropy(self):
        return tf.reduce_mean([tf.reduce_mean(layer.controller.entropy()) for layer in self.layers])

    def batch_selection_entropy(self):
        def layer_entropy(layer):
            probs = tf.reduce_mean(layer.controller.probs, axis=0)
            return -tf.reduce_sum(probs * tf.log(probs + 1e-30), axis=-1)
        return tf.reduce_mean([layer_entropy(layer) for layer in self.layers])

    def selection_logprob(self):
        with tf.variable_scope('selection_logprob'):
            def layer_selection_logprob(layer):
                logprobs = tf.log(layer.probs + 1e-20)
                log1probs = tf.log(1 - layer.probs + 1e-20)
                sel = tf.cast(layer.selection, tf.float32)
                term_1 = tf.multiply(logprobs, sel)
                term_2 = tf.multiply(log1probs, 1 - sel)
                return term_1 + term_2
            x = [tf.reduce_sum(
                layer_selection_logprob(layer), axis=-1) for layer in self.layers]
            return tf.reduce_sum(x, axis=0)

    def update_best_selection(self, best_selection_indices):
        """
        Args:
            best_selection_indices; size = Batch
        """
        def update(layer):
            """
            Args: 
                layer; named tuple of form ModularLayerAttributes
            """
            selection = tf.reshape(layer.selection, [self.sample_size, -1] + 
                                   layer.selection.shape[1:].as_list()) #[sample x B x parallel]
            new_best_selection = tensor_utils.gather_each(selection, best_selection_indices)
            return tf.scatter_update(layer.best_selection, self.data_indices, new_best_selection)
        return tf.group(*(update(layer) for layer in self.layers))

    def get_controller(self):
        """
        Returns the controller for visualisation purposes
        """
        return [layer.controller for layer in self.layers]

    def get_naive_kl(self):
        regulariser = tf.distributions.Bernoulli(0.3)
        def get_layer_kl(lay_number):
            ctrl = self.layers[lay_number].controller
            return tf.distributions.kl_divergence(ctrl, regulariser)
        return tf.reduce_sum([get_layer_kl(i) for i in range(len(self.layers))])

    def get_variational_kl(self, alpha, beta):
        def get_layer_KL(number):
            a = self.layers[number].a
            b = self.layers[number].b
            term_1 = tf.divide(- b + 1, b + 1e-20)
            term_2 = tf.log( tf.divide(tf.multiply(a, b), alpha + 1e-20) + 1e-20)
            term_bracket = (tf.digamma(1.) - tf.digamma(b) - tf.divide(1., b + 1e-20))
            term_3 = tf.multiply(tf.divide(a - alpha, a + 1e-20), term_bracket)
            return  (beta) * tf.reduce_sum(term_1 + term_2 + term_3)
        return tf.reduce_sum([get_layer_KL(i) for i in range(len(self.layers))])

    def get_kumaraswamy_logprob(self):
        def get_layer_logprob(number):
            a = self.layers[number].a
            b = self.layers[number].b
            pi = tf.check_numerics(self.layers[number].probs, 'pi')
            term_1 = tf.check_numerics(tf.log(tf.maximum(a,1e-20)) + tf.log(tf.maximum(b,1e-20)),'term_1')
            term_2 = tf.check_numerics(tf.multiply(a-1, tf.log(tf.maximum(pi, 1e-20))),'term_2')
            term_pi = tf.check_numerics(tf.pow(tf.maximum(pi, 1e-20), a), 'term_pi')
            term_log = tf.check_numerics(tf.log(tf.maximum(term_pi, 1e-20)), 'term_log')
            term_3 = tf.check_numerics(tf.multiply(b-1, term_log),'term_3')
            return tf.reduce_sum(term_1 + term_2 + term_3)
            # kum = tfd.Kumaraswamy(a, b)
            # return kum.log_prob(pi)
        return tf.reduce_sum([get_layer_logprob(i) for i in range(len(self.layers))])

    def get_pi_logprob(self, alpha):
        beta_param = 1.
        def get_layer_logprob(number):
            pi = tf.check_numerics(self.layers[number].probs, 'logpi:pi')
            norm = tf.lgamma(alpha) + tf.lgamma(beta_param) - tf.lgamma(alpha+beta_param)
            term_1 = tf.check_numerics(tf.multiply(tf.maximum(alpha-1,1e-20), tf.log(tf.maximum(pi, 1e-20))), 'term_1 logpi')
            term_2 = tf.check_numerics(tf.multiply(tf.maximum(beta_param-1,1e-20), tf.log(tf.maximum(1-pi, 1e-20))), 'term_2 logpi')
            return tf.reduce_sum(term_1 + term_2 - norm)
            # beta_dist = tf.distributions.Beta(alpha, beta_param)
            # return beta_dist.log_prob(pi)
        return tf.reduce_sum([get_layer_logprob(i) for i in range(len(self.layers))])


def run_modules(inputs, selection, module_fnc, output_shape):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()   
    #Used modules is just a list of modules that we are using
    used_modules, _ = tf.unique(tf.reshape(selection, (-1,))) #Size = No. of Modules

    def compute_module(accum, module):
        mask = tf.equal(module, selection) #select all the elements with the module we are using

        #OR operation on parallel axis, so that the input is passed through the module if any of the parallel has selected it
        reduced_mask = tf.reduce_any(mask, axis=-1) 

        indices = tf.where(reduced_mask) #Coordinates of TRUE
        affected_inp = tf.boolean_mask(inputs, reduced_mask) #Selects the batches that will go through this module
        output = module_fnc(affected_inp, module)

        #Add the outputs, scatter_nd makes it the right shape with 0s for inputs not computed
        return accum + tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64)) 

    output = tf.scan(compute_module, used_modules, initializer=tf.zeros(output_shape))[-1] #Want the last output of the scan fucntion
    return output #[sample * B x 10 (=units)]


def run_modules_withloop(inputs, selection, mask, module_fnc, output_shape):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()   
    #Used modules is just a list of modules that we are using
    used_modules, _ = tf.unique(tf.reshape(selection, (-1,))) #Size = No. of Modules

    condition = lambda accum, used_module, i: tf.less(i, tf.shape(used_modules)[0])

    def compute_module(accum, used_module, i):
        module = tf.slice(used_module, [i], [1])[0]

        equal_mask = tf.equal(module, selection) #select all the elements with the module we are using

        #OR operation on parallel axis, so that the input is passed through the module if any of the parallel has selected it
        reduced_mask = tf.reduce_any(equal_mask, axis=-1) 

        indices = tf.where(reduced_mask) #Coordinates of TRUE
        affected_inp = tf.boolean_mask(inputs, reduced_mask) #Selects the batches that will go through this module
        output = module_fnc(affected_inp, module, mask)

        #Add the outputs, scatter_nd makes it the right shape with 0s for inputs not computed
        full_output =  accum + tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64)) 

        i = tf.add(i, 1)
        return full_output, used_modules, i

    i = tf.constant(0, tf.int32)
    output = tf.while_loop(condition, compute_module, [tf.zeros(output_shape), used_modules, i])[0]

    return output



def run_masked_modules(inputs, selection, module_fnc, output_shape, ):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()
    #Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    def compute_module(accum, module):

        inputs_considered = tf.slice(selection, [0, module], [batch_size,1])
        mask = tf.reshape(tf.equal(1, inputs_considered), [-1])
        indices = tf.where(mask)
        affected_inp = tf.boolean_mask(inputs, mask)

        output = module_fnc(affected_inp, module)

        #Add the outputs, scatter_nd makes it the right shape with 0s for inputs not computed
        return accum + tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64)) 
    output = tf.scan(compute_module, used_modules, initializer=tf.zeros(output_shape))[-1] #Want the last output of the scan fucntion

    return output

def run_masked_modules_withloop(
    inputs, selection, mask, module_count,
    units, module_fnc, output_shape, weight, bias):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0, mask)
        output_shape = [batch_size] + dummy.shape[1:].as_list()

    #Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    condition = lambda accum, used_module, i: tf.less(i, tf.shape(used_modules)[0])

    def compute_module(accum, used_module, i):

        modules = tf.slice(selection, [0, i], [tf.shape(selection)[0], 1]) 
        input_mask = tf.reshape(tf.equal(1., modules), [-1])
        indices = tf.where(input_mask)
        affected_inp = tf.boolean_mask(inputs, input_mask)
        select_mask = tf.boolean_mask(mask, input_mask)
        select_weight = tf.boolean_mask(weight, input_mask)
        select_bias = tf.boolean_mask(bias, input_mask)


        output = module_fnc(affected_inp, i, select_mask, 
                            select_weight, select_bias)

        #Add the outputs, scatter_nd makes it the right shape with 0s for inputs not computed
        full_output =  accum + tf.scatter_nd(indices, 
                                            output, 
                                            tf.cast(output_shape, tf.int64)) 

        i = tf.add(i, 1)
        return full_output, used_modules, i

    i = tf.constant(0, tf.int32)
    output = tf.while_loop(
        condition, compute_module, [tf.zeros(output_shape), used_modules, i])[0]

    return output

def run_masked_modules_withloop_and_concat(
    inputs, selection, mask, module_count,
    units, module_fnc, output_shape, weight, bias):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0, mask, weight, bias)
        output_shape = [batch_size] + dummy.shape[1:].as_list()

    #Used modules is just a list of modules that we are using
    # used_modules = get_unique_modules(selection)

    condition = lambda accum, selection, i: tf.less(i, 
                                                    module_count)

    output_array = tf.TensorArray(dtype=tf.float32,
                                size=module_count)

    def compute_module(accum, selection, i):
        modules = tf.slice(selection, [0, i], [tf.shape(selection)[0], 1]) 
        input_mask = tf.reshape(tf.equal(1., modules), [-1])
        indices = tf.where(input_mask)

        affected_inp = tf.boolean_mask(inputs, input_mask)
        select_mask = tf.boolean_mask(mask, input_mask)
        select_weight = tf.boolean_mask(weight, input_mask)
        select_bias = tf.boolean_mask(bias, input_mask)


        output = module_fnc(affected_inp, i, select_mask, 
                            select_weight, select_bias)

        #Add the outputs, scatter_nd makes it the right shape 
        #with 0s for inputs not computed

        scatter = tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64))

        accum_write = accum.write(i, scatter)

        i = tf.add(i, 1)
        return accum_write, selection, i

    i = tf.constant(0, tf.int32)
    output = tf.while_loop(
        condition, compute_module, [output_array, selection, i])[0]

    full_output = output.stack()

    if full_output.shape.ndims>3:
        full_output = tf.transpose(full_output, [1,2,3,0,4])
        full_output = tf.reshape(full_output,
                                [tf.shape(full_output)[0],
                                dummy.shape[1].value,
                                dummy.shape[2].value,
                                units * module_count])
    else:
        full_output = tf.transpose(full_output, [1,0,2])
        full_output = tf.reshape(full_output,
                        [tf.shape(full_output)[0],
                        units * module_count])


    return full_output

def run_non_modular(
    inputs, selection, mask, module_count,
    units, module_fnc, output_shape, weight, bias):
    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0, mask, weight, bias)
        output_shape = [batch_size] + dummy.shape[1:].as_list()

    #Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    condition = lambda accum, selection, i: tf.less(i, 
                                                    module_count)

    output_array = tf.TensorArray(dtype=tf.float32,
                                size=module_count)

    def compute_module(accum, selection, i):
        modules = tf.slice(selection, [0, i], [tf.shape(selection)[0], 1]) 
        modules = tf.reshape(modules, [-1])
        # input_mask = tf.reshape(tf.equal(1., modules), [-1])
        # indices = tf.where(input_mask)
        # affected_inp = tf.boolean_mask(inputs, input_mask)
        output = module_fnc(inputs, i, mask, weight, bias)

        #Add the outputs, scatter_nd makes it the right shape 
        #with 0s for inputs not computed

        # scatter = tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64))

        accum_write = accum.write(i, output)

        i = tf.add(i, 1)
        return accum_write, selection, i

    i = tf.constant(0, tf.int32)
    output = tf.while_loop(
        condition, compute_module, [output_array, selection, i])[0]

    full_output = output.stack()

    if full_output.shape.ndims>3:
        full_output = tf.transpose(full_output, [1,2,3,0,4])
        full_output = tf.reshape(full_output,
                                [tf.shape(full_output)[0],
                                dummy.shape[1].value,
                                dummy.shape[2].value,
                                units * module_count])
    else:
        full_output = tf.transpose(full_output, [1,0,2])
        full_output = tf.reshape(full_output,
                        [tf.shape(full_output)[0],
                        units * module_count])
    return full_output

def e_step(template, sample_size, dataset_size, data_indices):
    context = ModularContext(ModularMode.E_STEP, data_indices, dataset_size, sample_size)

    # batch_size * sample_size
    loglikelihood = template(context)[0]
    assert loglikelihood.shape.ndims == 1

    # batch_size * sample_size
    selection_logprob = context.selection_logprob()
    assert selection_logprob.shape.ndims == 1

    logprob = tf.reshape(loglikelihood + selection_logprob, [sample_size, -1])
    best_selection_indices = tf.stop_gradient(tf.argmax(logprob, axis=0))

    return context.update_best_selection(best_selection_indices)


def m_step(
    template, optimizer, dataset_size, 
    data_indices, variational, num_batches, beta, sample_size, iteration):

    context = ModularContext(ModularMode.M_STEP, data_indices, dataset_size, sample_size)

    if variational == 'False':
        print('NOT VAR')
        loglikelihood = template(context)[0]  
        selection_logprob = context.selection_logprob()
        KL = context.get_variational_kl(0.3)

        ctrl_objective = -tf.reduce_mean(selection_logprob)
        module_objective = -tf.reduce_mean(loglikelihood)
        joint_objective =  -(tf.reduce_mean(selection_logprob + loglikelihood - KL))

        tf.summary.scalar('KL', tf.reduce_sum(KL), collections=[M_STEP_SUMMARIES])
        tf.summary.scalar('ctrl_objective', ctrl_objective, collections=[M_STEP_SUMMARIES])
        tf.summary.scalar('module_objective', module_objective, collections=[M_STEP_SUMMARIES])
        tf.summary.scalar('ELBO', -joint_objective, collections=[M_STEP_SUMMARIES])
    else:
        print('VAR')
        loglikelihood = template(context)[0]
        # log_pi = context.get_pi_logprob(0.005)
        # log_kum = context.get_kumaraswamy_logprob()
        # log_pi = tf.check_numerics(log_pi, 'logpi')
        # log_kum = tf.check_numerics(log_kum, 'log_kum')
        # score_term = (beta/num_batches) * (log_pi - log_kum)

        # joint_objective = - (tf.reduce_sum(loglikelihood) + score_term)


        damp = get_damper(iteration, get_damp_list(num_batches))

        KL = context.get_variational_kl(0.05, beta)
        mod_KL = tf.reduce_sum((1/num_batches) * KL)

        joint_objective = - (loglikelihood - mod_KL)

        tf.summary.scalar('KL', mod_KL, collections=[M_STEP_SUMMARIES])
        tf.summary.scalar('ELBO', -joint_objective, collections=[M_STEP_SUMMARIES])
        module_objective =  tf.reduce_sum(loglikelihood)
        tf.summary.scalar('module_objective', -module_objective, collections=[M_STEP_SUMMARIES])
        # tf.summary.scalar('damp', tf.reduce_sum(damp), collections=[M_STEP_SUMMARIES])


        tf.add_to_collection(name='mod_KL',
                            value=mod_KL)
        # tf.add_to_collection(name='Damp',
        #                     value=damp)
        tf.add_to_collection(name='KL',
                            value=KL)

    # with tf.control_dependencies([moving_average]):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = optimizer.minimize(joint_objective)

    return opt

def get_damper(iteration, damp_list):
    return tf.slice(damp_list, [tf.cast(iteration, tf.int32)], [1])

def get_damp_list(num_batches):
    iteration = tf.range(num_batches)
    term_1 = (num_batches-iteration)*tf.log(2.)
    term_2 = num_batches*tf.log(2.)
    damp = tf.exp(term_1 - term_2)
    damp = damp/tf.reduce_sum(damp)
    return tf.reverse(damp, axis=[0])

def evaluation(template, data_indices, dataset_size):
    context = ModularContext(ModularMode.EVALUATION, data_indices, dataset_size)
    return template(context)

def get_unique_modules(selection):
    ones = tf.equal(1.,selection)
    b,m = tf.shape(ones)[0],  tf.shape(ones)[1]
    modules_idx = tf.range(m)
    tiled = tf.tile(modules_idx, [b])
    tile_re = tf.reshape(tiled,[b,m])
    mask = tf.boolean_mask(tile_re, ones)
    uniq, _ = tf.unique(mask)
    return uniq

def create_m_step_summaries():
    return tf.summary.merge_all(key=M_STEP_SUMMARIES)

def get_tensor_op():
    return tf.get_collection('mod_KL')

def get_op():
    return tf.get_collection('Damp')

def get_KL():
    return tf.get_collection('KL')

