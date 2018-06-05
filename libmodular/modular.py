from collections import namedtuple
from enum import Enum
from typing import List
import libmodular.tensor_utils as tensor_utils

import tensorflow as tf

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
            return tf.tile(inputs, [self.sample_size] +[1] * (rank - 1)) #2D arg for tile for the 2 axes!
        return inputs

    def selection_entropy(self):
        return tf.reduce_mean([tf.reduce_mean(layer.controller.entropy()) for layer in self.layers])

    def batch_selection_entropy(self):
        def layer_entropy(layer):
            probs = tf.reduce_mean(layer.controller.probs, axis=0) #[parallel x module]
            reduce_prob = tf.reduce_sum(-tf.reduce_sum(probs * tf.log(probs + 1e-30), axis=-1))
            return reduce_prob
        hb = tf.reduce_mean([layer_entropy(layer) for layer in self.layers])
        return hb

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


def run_modules(inputs, selection, module_fnc, output_shape):

    batch_size = tf.shape(inputs)[0]
    output_shape = [batch_size] + output_shape
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
    output_shape = [batch_size] + output_shape
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


def e_step(template, sample_size, dataset_size, data_indices):
    #Initialise Modular Context here
    context = ModularContext(ModularMode.E_STEP, data_indices, dataset_size, sample_size)

    #Get log likelihood from the network function
    #log likelihood and selection log prob differ as one is with the logits of the controller layer
    #and one is with the logits of the module layer
    loglikelihood = template(context)[0]
    selection_logprob = context.selection_logprob() #Log prob of selection

    shape = [sample_size, -1] + loglikelihood.shape[1:].as_list() #Not sure why 3rd term is needed (for parallel)
    logprob = tf.reshape(loglikelihood + selection_logprob, shape) #[sample x B]
    best_selection_indices = tf.stop_gradient(tf.argmax(logprob, axis=0)) #[B]

    return context.update_best_selection(best_selection_indices)


def m_step(template, optimizer, dataset_size, data_indices):
    context = ModularContext(ModularMode.M_STEP, data_indices, dataset_size)

    loglikelihood = template(context)[0]
    selection_logprob = context.selection_logprob()

    ctrl_objective = -tf.reduce_mean(selection_logprob)
    module_objective = -tf.reduce_mean(loglikelihood)

    return optimizer.minimize(ctrl_objective + module_objective)


def evaluation(template):
    context = ModularContext(ModularMode.EVALUATION)
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
