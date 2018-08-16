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
                                        'units'])


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

    def get_beta_logprob(self):
        def _layer_logprob(number):
            a = tf.check_numerics(self.layers[number].a, 'a') + 1e-20
            b = tf.check_numerics(self.layers[number].b, 'b') + 1e-20
            pi = tf.check_numerics(self.layers[number].probs, 'pi') + 1e-20
            n_a = a
            n_b = b
            term_norm = tf.lgamma(n_a) + tf.lgamma(n_b) - tf.lgamma(n_a + n_b)
            term_1 = tf.multiply(n_a-1, tf.log(pi))
            term_2 = tf.multiply(n_b-1, tf.log(1-pi+1e-20))
            return tf.reduce_sum(tf.reduce_mean(term_1 + term_2 - term_norm, axis=0))
        return tf.reduce_sum([_layer_logprob(i) for i in range(len(self.layers))])


    def control_variate(self, w):
        def _layer_logprob(number):
            a = tf.check_numerics(self.layers[number].a, 'a') + 1e-20
            b = tf.check_numerics(self.layers[number].b, 'b') + 1e-20
            pi = tf.check_numerics(self.layers[number].probs, 'pi') + 1e-20
            term_norm = tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a + b)
            term_1 = tf.multiply(a-1, tf.log(pi))
            term_2 = tf.multiply(b-1, tf.log(1-pi+1e-20))
            return tf.reduce_sum(w * tf.reduce_mean(term_1 + term_2 - term_norm, axis=0))
            # return tf.check_numerics(tf.distributions.Beta(a,b).log_prob(pi), 'beta again')
        return tf.reduce_sum([_layer_logprob(i) for i in range(len(self.layers))])


    def get_prior_beta(self, alpha):
        beta = tf.constant(1.)
        def _layer_logprob(number):
            pi = tf.check_numerics(self.layers[number].probs, 'pi') + 1e-20
            z = tf.cast(self.layers[number].selection, tf.float32)
            n_alpha = tf.constant(alpha)
            n_beta = beta
            term_norm = tf.lgamma(n_alpha) + tf.lgamma(n_beta) - tf.lgamma(n_alpha + n_beta)
            term_1 = tf.multiply(n_alpha-1, tf.log(pi+1e-20))
            term_2 = tf.multiply(n_beta-1, tf.log(1-pi+1e-20))
            return tf.reduce_sum(tf.reduce_mean(term_1 + term_2 - term_norm, axis=0))
            # return tf.check_numerics(tf.distributions.Beta(a,b).log_prob(pi), 'beta again')
        return tf.reduce_sum([_layer_logprob(i) for i in range(len(self.layers))])

    def get_naive_kl(self):
        regulariser = tf.distributions.Bernoulli(0.3)
        def get_layer_kl(lay_number):
            ctrl = self.layers[lay_number].controller
            return tf.distributions.kl_divergence(ctrl, regulariser)
        return tf.reduce_sum([get_layer_kl(i) for i in range(len(self.layers))])

    def get_variational_kl(self, alpha):
        def get_layer_KL(number):
            a = self.layers[number].a
            b = self.layers[number].b
            term_1 = tf.divide(- b + 1, b + 1e-20)
            term_2 = tf.log( tf.divide(tf.multiply(a, b), alpha + 1e-20) + 1e-20)
            term_bracket = (tf.digamma(1.) - tf.digamma(b) - tf.divide(1., b + 1e-20))
            term_3 = tf.multiply(tf.divide(a - alpha, a + 1e-20), term_bracket)
            return tf.reduce_sum(term_1 + term_2 + term_3)
        return tf.reduce_sum([get_layer_KL(i) for i in range(len(self.layers))])

    def get_bern_prior(self):
        p = tf.constant(0.3)
        def get_layer_prior(number):
            z = self.layers[number].selection
            return tfd.Bernoulli(probs=p).log_prob(z)
        return tf.reduce_sum([get_layer_prior(i) for i in range(len(self.layers))])
    
    def get_bern_logprob(self):
        def get_layer(number):
            ctrl = self.layers[number].controller
            z = self.layers[number].selection
            return tf.reduce_sum(ctrl.log_prob(z))
        return tf.reduce_sum([get_layer(i) for i in range(len(self.layers))])

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
    units, module_fnc, output_shape):

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

        module = tf.slice(used_module, [i], [1])
        inputs_considered = tf.slice(selection, 
                                    [0, module[0]], 
                                    [batch_size, 1])
        re_mask = tf.reshape(tf.equal(1.,inputs_considered), [-1])
        indices = tf.where(re_mask)
        affected_inp = tf.boolean_mask(inputs, re_mask)

        output = module_fnc(affected_inp, module[0], mask)

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
    units, module_fnc, output_shape):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0, mask)
        output_shape = [batch_size] + dummy.shape[1:].as_list()


    #Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    condition = lambda accum, selection, i: tf.less(i, 
                                                    module_count)

    output_array = tf.TensorArray(dtype=tf.float32,
                                size=module_count)

    def compute_module(accum, selection, i):
        modules = tf.slice(selection, [0, i], [tf.shape(selection)[0], 1]) 
        input_mask = tf.reshape(tf.equal(1, modules), [-1])
        indices = tf.where(input_mask)

        affected_inp = tf.boolean_mask(inputs, input_mask)
        output = module_fnc(affected_inp, i, mask)

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
    units, module_fnc, output_shape):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0, mask)
        output_shape = [batch_size] + dummy.shape[1:].as_list()


    #Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    condition = lambda accum, selection, i: tf.less(i, 
                                                    module_count)

    output_array = tf.TensorArray(dtype=tf.float32,
                                size=module_count)

    def compute_module(accum, selection, i):
        modules = tf.slice(selection, [0, i], [tf.shape(selection)[0], 1]) 
        # input_mask = tf.reshape(tf.equal(1, modules), [-1])
        # indices = tf.where(input_mask)

        # affected_inp = tf.boolean_mask(inputs, input_mask)
        output = module_fnc(inputs, i, mask)

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


def e_step(template, dataset_size, data_indices, opt):
    context = ModularContext(ModularMode.E_STEP, data_indices, dataset_size)

    # batch_size * sample_size
    loglikelihood = template(context)[0]

    module_objective =  tf.reduce_sum(loglikelihood) + context.get_prior_beta(0.05)

    kum_log = context.get_beta_logprob()

    path_term = tf.stop_gradient(module_objective - kum_log)

    joint_objective = tf.multiply(kum_log, path_term)

    tf.summary.scalar('E step objective', joint_objective, collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('E module_objective', module_objective, collections=[M_STEP_SUMMARIES])

    return opt.minimize(-joint_objective)


def m_step(
    template, optimizer, dataset_size, 
    data_indices, num_batches, sample_size):
    context = ModularContext(ModularMode.M_STEP, data_indices, dataset_size, sample_size)

    print('REINFORCE')
    loglikelihood = template(context)[0]
    
    module_objective =  loglikelihood + context.get_prior_beta(0.05)

    q_log = context.get_beta_logprob()

    path_term = tf.stop_gradient(module_objective - q_log) - context.control_variate(10)

    E_joint_objective = tf.multiply(q_log, path_term)

    M_joint_objective = module_objective

    total_objective = E_joint_objective + M_joint_objective

    tf.summary.scalar('M step objective', M_joint_objective, collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('E step objective', E_joint_objective, collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('module_objective', module_objective, collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('ELBO', -total_objective, collections=[M_STEP_SUMMARIES])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = optimizer.minimize(-total_objective)

    return opt


def evaluation(template, data_indices, dataset_size):
    context = ModularContext(ModularMode.EVALUATION, data_indices, dataset_size)
    return template(context)

def get_unique_modules(selection):
    ones = tf.equal(1,selection)
    b,m = tf.shape(ones)[0],  tf.shape(ones)[1]
    modules_idx = tf.range(m)
    tiled = tf.tile(modules_idx, [b])
    tile_re = tf.reshape(tiled,[b,m])
    mask = tf.boolean_mask(tile_re, ones)
    uniq, _ = tf.unique(mask)
    return uniq

def create_m_step_summaries():
    return tf.summary.merge_all(key=M_STEP_SUMMARIES)