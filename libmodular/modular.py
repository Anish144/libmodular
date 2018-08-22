from collections import namedtuple
from enum import Enum
from typing import List
import libmodular.tensor_utils as tensor_utils

import tensorflow as tf

M_STEP_SUMMARIES = 'M_STEP_SUMMARIES'
ModularMode = Enum('ModularMode', 'E_STEP M_STEP EVALUATION')
ModularLayerAttributes = namedtuple('ModularLayerAttributes', ['selection', 'best_selection', 'controller'])
ModulePool = namedtuple('ModulePool', ['module_count', 'module_fnc', 'output_shape'])


class ModularContext:

    def __init__(self, mode: ModularMode, data_indices=None, dataset_size: int = None, sample_size: int = 1):
        self.mode = mode
        self.data_indices = data_indices
        self.dataset_size = dataset_size
        self.sample_size = sample_size
        self.e_step_samples = False
        self.layers: List[ModularLayerAttributes] = [] #Save the module layer attributes in "Modular_Layer"

    def begin_modular(self, inputs):
        if self.mode == ModularMode.E_STEP and not self.e_step_samples:
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
        x = [tf.reduce_sum(attrs.controller.log_prob(attrs.selection), axis=-1) for attrs in self.layers]
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
            selection = tf.reshape(layer.selection, [self.sample_size, -1] + layer.selection.shape[1:].as_list()) #[sample x B x parallel]
            new_best_selection = tensor_utils.gather_each(selection, best_selection_indices)
            return tf.scatter_update(layer.best_selection, self.data_indices, new_best_selection)
        return tf.group(*(update(layer) for layer in self.layers))

    def get_controller(self):
        """
        Returns the controller for visualisation purposes
        """
        return [layer.controller for layer in self.layers]

    def get_kl(self):
        regulariser = tf.distributions.Bernoulli(0.3)
        def get_layer_kl(lay_number):
            ctrl = self.layers[lay_number].controller
            return tf.distributions.kl_divergence(ctrl, regulariser)
        return tf.reduce_sum([get_layer_kl(i) for i in range(len(self.layers))])



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


def run_masked_modules(inputs, selection, module_fnc, output_shape):

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


def run_masked_modules_withloop(inputs, selection, mask, module_fnc, output_shape):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()

    #Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    condition = lambda accum, used_module, i: tf.less(i, tf.shape(used_modules)[0])

    def compute_module(accum, used_module, i):

        module = tf.slice(used_module, [i], [1])
        inputs_considered = tf.slice(selection, 
                                    [0, module[0]], 
                                    [batch_size, 1])
        re_mask = tf.reshape(tf.equal(1,inputs_considered), [-1])
        indices = tf.where(re_mask)
        affected_inp = tf.boolean_mask(inputs, re_mask)

        output = module_fnc(affected_inp, module[0])

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


def m_step(template, optimizer, dataset_size, data_indices):
    context = ModularContext(ModularMode.M_STEP, data_indices, dataset_size)
    loglikelihood = template(context)[0]
    selection_logprob = context.selection_logprob()

    ctrl_objective = -tf.reduce_mean(selection_logprob)
    module_objective = -tf.reduce_mean(loglikelihood)
    joint_objective = ctrl_objective + module_objective + context.get_kl()

    tf.summary.scalar('ctrl_objective', ctrl_objective, collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('module_objective', module_objective, collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('joint_objective', joint_objective, collections=[M_STEP_SUMMARIES])

    return optimizer.minimize(joint_objective)


def evaluation(template, data_indices):
    context = ModularContext(ModularMode.EVALUATION, data_indices)
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