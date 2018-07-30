import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib as tfc

class Model:

	def __init__(self, inp, target, batch_size, target_num_classes, is_training,
				 hyperparams):
		self.input = inp
		self.target = target
		self.batch_size = batch_size
		self.target_num_classes = target_num_classes
		self.is_training = is_training
		self.hp = hyperparams

	def embedded_input(self):
		params = tf.get_variable('params', (self.target_num_classes, self.hp.embedding_size))
		embedding = tf.nn.embedding_lookup(params, self.input)
		return embedding

	def logits(self):
		with tf.variable_scope('logits', initializer=self.initializer):
			return tf.layers.dense(self.rnn_output, self.target_num_classes)

	def loss(self):
		distr = tfd.Categorical(self.logits)
		loss = -tf.reduce_mean(distr.log_prob(self.target))
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			return tf.identity(loss)

def new_controller(self, initializer):
    with tf.variable_scope(None, 'New_controller'):

        inputs = context.begin_modular(inputs)
        flat_inputs = tf.layers.flatten(inputs)

        shape = modules.module_count
        a = tf.get_variable(name='a', dtype=tf.float32, initializer=tf.random_uniform([shape], 
                                                                                       minval=3.8, maxval=4.2))
        b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.random_uniform([shape], 
                                                                                      minval=1.8, maxval=2.2))
        u = get_u(shape)
        pi = tf.pow((1 - tf.pow(u, tf.divide(1,b + 1e-20))), tf.divide(1,a + 1e-20))
        
        input_shape = flat_inputs.shape[-1].value

        eta = tf.get_variable(name='eta', shape=[shape], dtype=tf.float32)
        khi = tf.get_variable(name='khi', shape=[shape], dtype=tf.float32)
        beta = tfd.MultivariateNormalDiag(eta, khi)

        gamma = tf.get_variable(name='gamma', shape=[shape], dtype=tf.float32)

        def dependent_pi(inputs, pi, gamma, beta, eps):
            with tf.variable_scope('dep_pi', reuse=tf.AUTO_REUSE):
                stop_input = tf.stop_gradient(inputs)
                reshape = tf.reshape(stop_input, 
                             [tf.shape(inputs)[0], shape, 
                             tf.cast(tf.divide(tf.shape(inputs)[-1], shape), 
                             tf.int32)])
                re_sum = tf.reduce_sum(reshape, axis=2)
                mean, var = tf.nn.moments(re_sum, axes=0, keep_dims=True)
                #Moving Average
                ema = tf.train.ExponentialMovingAverage(decay=0.5,
                                                        zero_debias=False)

                ema_opt = ema.apply(var_list=[mean, var])
                mean_avg = ema.average(mean)
                var_avg = ema.average(var)

                new_input = tf.stop_gradient(tf.nn.batch_normalization(re_sum,
                                                                       mean_avg,
                                                                       var_avg,
                                                                       beta,
                                                                       gamma,
                                                                       0.0001
                                                        ))
                max_input = tf.maximum(eps, new_input)
                dep = tf.minimum(1-eps, max_input) + 1e-20

                return tf.multiply(dep, pi), dep, ema_opt


        rho = 5.17
        beta_prior = tfd.MultivariateNormalDiag(tf.zeros([shape]), 
                                                tf.multiply(rho, tf.ones([shape]))
                                               )

        logits, dep, ema = dependent_pi(flat_inputs, pi, gamma, beta.sample(), eps=0.0001)

        ctrl_bern = tfd.Bernoulli(probs=logits)

        best_shape = [context.dataset_size, modules.module_count]
        best_selection_persistent = tf.get_variable('best_selection', shape=best_shape, dtype=tf.int32, 
                                                    initializer=initializer)

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

        attrs = ModularLayerAttributes(selection, best_selection_persistent, 
                                       ctrl_bern, a, b, beta, beta_prior)
        context.layers.append(attrs) 

        return (run_masked_modules_withloop(inputs, selection, 
                                           modules.module_fnc, 
                                           modules.output_shape), 
               ema, logits, pi)