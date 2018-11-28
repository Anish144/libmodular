from collections import namedtuple
from enum import Enum
from typing import List
import libmodular.tensor_utils as tensor_utils

import tensorflow as tf

M_STEP_SUMMARIES = 'M_STEP_SUMMARIES'
ModularMode = Enum('ModularMode', 'E_STEP M_STEP EVALUATION')
ModularLayerAttributes = namedtuple(
    'ModularLayerAttributes',
    ['selection', 'best_selection',
     'controller'])
ModulePool = namedtuple(
    'ModulePool',
    ['module_count', 'module_fnc', 'output_shape', 'units', 'weight', 'bias'])


class ModularContext:

    def __init__(self, mode: ModularMode, data_indices=None, dataset_size: int = None, sample_size: int = 1):
        self.mode = mode
        self.data_indices = data_indices
        self.dataset_size = dataset_size
        self.sample_size = sample_size
        self.e_step_samples = False
        # Save the module layer attributes in "Modular_Layer"
        self.layers: List[ModularLayerAttributes] = []

    def begin_modular(self, inputs):
        if self.mode == ModularMode.E_STEP and not self.e_step_samples:
            self.e_step_samples = True
            rank = inputs.shape.ndims
            return tf.tile(inputs, [self.sample_size] + [1] * (rank - 1))
        return inputs

    def selection_entropy(self):
        return tf.reduce_mean([tf.reduce_mean(layer.controller.entropy()) for layer in self.layers])

    def batch_selection_entropy(self):
        def layer_entropy(layer):
            probs = tf.reduce_mean(layer.controller.probs, axis=0)
            return -tf.reduce_sum(probs * tf.log(probs + 1e-30), axis=-1)
        return tf.divide(tf.reduce_sum([tf.reduce_sum(layer_entropy(layer)) for layer in self.layers]), tf.constant(len(self.layers), tf.float32))

    # def selection_logprob(self):
    #     def layer_logprob(layer):
    #         probs = self.layers[layer].probs
    #         selection = tf.cast(self.layers[layer].selection, tf.float32)
    #         term_1 = selection * tf.log(tf.maximum(probs, 1e-20))
    #         term_2 = (1-selection) * tf.log(tf.maximum(1-probs, 1e-20))
    #         return term_1 + term_2
    #     x = [tf.reduce_sum(layer_logprob(attrs), axis=-1) for attrs in range(len(self.layers))]
    #     return tf.reduce_sum(x, axis=0)

    def selection_logprob(self):
        x = [tf.reduce_sum(attrs.controller.log_prob(
            attrs.selection), axis=-1) for attrs in self.layers]
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
            selection = tf.reshape(layer.selection, [
                                   self.sample_size, -1] + layer.selection.shape[1:].as_list())  # [sample x B x parallel]
            new_best_selection = tensor_utils.gather_each(
                selection, best_selection_indices)
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
    # Used modules is just a list of modules that we are using
    used_modules, _ = tf.unique(tf.reshape(
        selection, (-1,)))  # Size = No. of Modules

    def compute_module(accum, module):
        # select all the elements with the module we are using
        mask = tf.equal(module, selection)

        # OR operation on parallel axis, so that the input is passed through the module if any of the parallel has selected it
        reduced_mask = tf.reduce_any(mask, axis=-1)

        indices = tf.where(reduced_mask)  # Coordinates of TRUE
        # Selects the batches that will go through this module
        affected_inp = tf.boolean_mask(inputs, reduced_mask)
        output = module_fnc(affected_inp, module)

        # Add the outputs, scatter_nd makes it the right shape with 0s for inputs not computed
        return accum + tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64))

    output = tf.scan(compute_module, used_modules, initializer=tf.zeros(
        output_shape))[-1]  # Want the last output of the scan fucntion
    return output  # [sample * B x 10 (=units)]


def run_modules_concat(inputs, selection, module_fnc, output_shape, module_count):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()
    # Used modules is just a list of modules that we are using
    used_modules, _ = tf.unique(tf.reshape(
        selection, (-1,)))  # Size = No. of Modules

    def condition(accum, module, i): return tf.less(i,
                                                    module_count)

    output_array = tf.TensorArray(dtype=tf.float32,
                                  size=module_count)

    def compute_module(accum, module, i):
        # select all the elements with the module we are using
        mask = tf.equal(module, selection)
        # OR operation on parallel axis, so that the input is passed through the module if any of the parallel has selected it
        reduced_mask = tf.reduce_any(mask, axis=-1)
        indices = tf.where(reduced_mask)  # Coordinates of TRUE
        # Selects the batches that will go through this module
        affected_inp = tf.boolean_mask(inputs, reduced_mask)
        output = module_fnc(affected_inp, module)

        scatter = tf.scatter_nd(
            indices, output, tf.cast(output_shape, tf.int64))
        accum_write = accum.write(i, scatter)

        i = tf.add(i, 1)
        return accum_write, module, i

    i = tf.constant(0, tf.int32)
    output = tf.while_loop(
        condition, compute_module, [output_array, used_modules, i])[0]

    full_output = output.stack()

    return full_output


def run_masked_modules(inputs, selection, module_fnc, output_shape):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()
    # Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    def compute_module(accum, module):

        inputs_considered = tf.slice(selection, [0, module], [batch_size, 1])
        mask = tf.reshape(tf.equal(1, inputs_considered), [-1])
        indices = tf.where(mask)
        affected_inp = tf.boolean_mask(inputs, mask)
        output = module_fnc(affected_inp, module)

        # Add the outputs, scatter_nd makes it the right shape with 0s for inputs not computed
        return accum + tf.scatter_nd(indices, output, tf.cast(output_shape, tf.int64))
    output = tf.scan(compute_module, used_modules, initializer=tf.zeros(
        output_shape))[-1]  # Want the last output of the scan fucntion
    return output


def run_masked_modules_withloop(
    inputs, selection, mask, module_count,
    units, module_fnc, output_shape
):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()

    # Used modules is just a list of modules that we are using
    used_modules = get_unique_modules(selection)

    def condition(accum, used_module, i): return tf.less(
        i, tf.shape(used_modules)[0])

    def compute_module(accum, used_module, i):

        module = tf.slice(used_module, [i], [1])
        inputs_considered = tf.slice(selection,
                                     [0, module[0]],
                                     [batch_size, 1])
        re_mask = tf.reshape(tf.equal(1, inputs_considered), [-1])
        indices = tf.where(re_mask)
        affected_inp = tf.boolean_mask(inputs, re_mask)

        output = module_fnc(affected_inp, module[0])

        output = tf.nn.relu(output)

        # Add the outputs, scatter_nd makes it the right shape with 0s for inputs not computed
        full_output = accum + tf.scatter_nd(indices,
                                            output,
                                            tf.cast(output_shape, tf.int64))

        i = tf.add(i, 1)
        return full_output, used_modules, i

    i = tf.constant(0, tf.int32)
    output = tf.while_loop(
        condition,
        compute_module,
        [tf.zeros(output_shape), used_modules, i])[0]

    # Need to average outputs of modules by the number used
    selection_cast = tf.cast(selection, tf.float32)
    summed_selection = tf.reduce_sum(selection_cast, axis=1)
    safe_selected = tf.maximum(summed_selection, 1e-20)
    inverted_summed_selection = tf.divide(1, safe_selected)
    safe_inverted = tf.maximum(inverted_summed_selection, 1e-20)

    if output.shape.ndims > 3:
        output = tf.einsum('bhwc,b->bhwc', output, safe_inverted)
    else:
        output = tf.einsum('bk,b->bk', output, safe_inverted)
    return output


def run_masked_modules_withloop_and_concat(
    inputs, selection, mask, module_count,
    units, module_fnc, output_shape, weight, bias
):

    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily
        dummy = module_fnc(inputs, 0, weight, bias)
        output_shape = [batch_size] + dummy.shape[1:].as_list()

    def condition(accum, selection, i): return tf.less(i,
                                                       module_count)

    output_array = tf.TensorArray(dtype=tf.float32,
                                  size=module_count)

    def compute_module(accum, selection, i):
        modules = tf.slice(selection, [0, i], [tf.shape(selection)[0], 1])
        input_mask = tf.reshape(tf.equal(1, modules), [-1])
        indices = tf.where(input_mask)

        affected_inp = tf.boolean_mask(inputs, input_mask)
        select_mask = tf.boolean_mask(mask, input_mask)
        select_weight = tf.boolean_mask(weight, input_mask)
        select_bias = tf.boolean_mask(bias, input_mask)

        output = module_fnc(affected_inp, i, select_weight, select_bias)

        # Add the outputs, scatter_nd makes it the right shape
        # with 0s for inputs not computed

        scatter = tf.scatter_nd(
            indices, output, tf.cast(output_shape, tf.int64))

        accum_write = accum.write(i, scatter)

        i = tf.add(i, 1)
        return accum_write, selection, i

    i = tf.constant(0, tf.int32)
    output = tf.while_loop(
        condition, compute_module, [output_array, selection, i])[0]

    full_output = output.stack()

    if full_output.shape.ndims > 3:
        full_output = tf.transpose(full_output, [1, 2, 3, 0, 4])
        full_output = tf.reshape(full_output,
                                 [tf.shape(full_output)[0],
                                  dummy.shape[1].value,
                                  dummy.shape[2].value,
                                  units * module_count])
    else:
        full_output = tf.transpose(full_output, [1, 0, 2])
        full_output = tf.reshape(full_output,
                                 [tf.shape(full_output)[0],
                                  units * module_count])

    return full_output


def e_step(template, sample_size, dataset_size, data_indices):
    context = ModularContext(
        ModularMode.E_STEP, data_indices, dataset_size, sample_size)

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
    joint_objective = ctrl_objective + module_objective

    tf.summary.scalar('ctrl_objective', ctrl_objective,
                      collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('module_objective', module_objective,
                      collections=[M_STEP_SUMMARIES])
    tf.summary.scalar('joint_objective', joint_objective,
                      collections=[M_STEP_SUMMARIES])

    return optimizer.minimize(joint_objective)


def evaluation(template, data_indices):
    context = ModularContext(ModularMode.EVALUATION, data_indices)
    return template(context)


def get_unique_modules(selection):
    ones = tf.equal(1, selection)
    b, m = tf.shape(ones)[0],  tf.shape(ones)[1]
    modules_idx = tf.range(m)
    tiled = tf.tile(modules_idx, [b])
    tile_re = tf.reshape(tiled, [b, m])
    mask = tf.boolean_mask(tile_re, ones)
    uniq, _ = tf.unique(mask)
    return uniq


def create_m_step_summaries():
    return tf.summary.merge_all(key=M_STEP_SUMMARIES)
