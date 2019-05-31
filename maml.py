""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize
import random

import os
import warnings

# Dependency imports
from absl import flags
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
#import keras.layers as kl
#from keras.layers.normalization import BatchNormalization
#from tensorflow_probability.python import distributions as tfd
'''
tfe = tf.contrib.eager
try:
    tfe.enable_eager_execution()
except ValueError:
    pass
'''

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

FLAGS = flags.FLAGS


class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=1):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [40, 40]
            self.loss_func = mse
#            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
            self.loss_func = xent
            self.classification = True

#   set up NN structure

            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
#                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
#                self.forward=self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
        else:
            raise ValueError('Unrecognized data source.')




    def construct_conv_weights(self):
        #with tf.name_scope("bayesian_neural_net", values=[images]):
        k=3
        channels = self.channels
        model = tf.keras.Sequential([
            #tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=k, padding="SAME", activation=tf.nn.relu,input_shape=(None,28, 28, 1)),
            tf.keras.layers.Reshape((self.img_size, self.img_size,channels)),
            tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=k, padding="SAME", activation=tf.nn.relu),
            #tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
            tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=5, padding="SAME", activation=tf.nn.relu),
            #tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
            tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=5, padding="SAME",activation=tf.nn.relu),
            #tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
            tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=5, padding="SAME",activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(84, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(self.dim_output)])

        return model


    def construct_fc_weights(self):
        #with tf.name_scope("bayesian_neural_net", values=[images]):
        model = tf.keras.Sequential()
        #model = tf.keras.Sequential([tfp.layers.DenseFlipout(self.dim_hidden[0],input_shape=(self.dim_input,) ,activation=tf.nn.relu,kernel_initializer='random_uniform')])
        for i in range(len(self.dim_hidden)):
            model.add(tfp.layers.DenseFlipout(self.dim_hidden[i] ,activation=tf.nn.relu))
            #model.add(tf.keras.layers.Dense(self.dim_hidden[i]))
            #model.add(tf.keras.layers.BatchNormalization())
        model.add(tfp.layers.DenseFlipout(self.dim_output))

        return model



    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if FLAGS.datasource == 'sinusoid':
        #if input_tensors is None:
            #print('fuck its none')
            input_shape = (FLAGS.meta_batch_size,FLAGS.update_batch_size,self.dim_input)
            #input_b_shape = (FLAGS.meta_batch_size,0,self.dim_input)
            output_shape = (FLAGS.meta_batch_size,FLAGS.update_batch_size,self.dim_output)
            #output_b_shape = (FLAGS.meta_batch_size,0,self.dim_output)
            self.inputa = tf.placeholder(tf.float32,shape=input_shape)
            self.inputb = tf.placeholder(tf.float32,shape=input_shape)
            self.labela = tf.placeholder(tf.float32,shape=output_shape)
            self.labelb = tf.placeholder(tf.float32,shape=output_shape)
            self.inputa_init = input_tensors['inputa_init']
            self.inputa_init = (self.inputa_init).astype('float32')
        else:
            #print('its not none fuck')
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']
            self.inputa_init = self.inputa
            self.inputa_init = tf.cast(self.inputa_init, tf.float32)

        self.sigma = FLAGS.sigma
        self.num_repeat = FLAGS.num_repeat

        N_task=FLAGS.meta_batch_size
        print('N_task=',N_task)

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print('reuse!!!')
                training_scope.reuse_variables()
                weights = self.weights


            else:
                print('it is fine')
                # Define the weights /  weights stands for the model nueral_net!!!!!!!
                # run models with array input to initialize models
                #random.seed(7)
                self.weights = weights = self.construct_weights()
                weights(self.inputa_init[0])

            weights_a = self.construct_weights()
            weights_a(self.inputa_init[0])
            weights_b = self.construct_weights()
            weights_b(self.inputa_init[0])
            weights_a_stop = self.construct_weights()
            weights_a_stop(self.inputa_init[0])
            weights_b_stop = self.construct_weights()
            weights_b_stop(self.inputa_init[0])
            weights_output = self.construct_weights()
            weights_output(self.inputa_init[0])


            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            #num_updates = max(self.test_num_updates, FLAGS.num_updates)
            if FLAGS.test_num_updates==-1:
                num_updates = FLAGS.num_updates
            else:
                num_updates = FLAGS.test_num_updates

            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates



            def predict(NN,inputs):
                logits=[]
                for i in range(self.num_repeat):
                    logits.append(NN(inputs))
                mean = tf.reduce_mean(logits,0)
                std = reduce_std(logits,0)
                return mean, std

            def deter(model_out,model):
                for i, layer in enumerate(model_out.layers):
                    try:
                        layer.kernel_posterior = tfd.Independent(tfd.Normal(loc=model.layers[i].kernel_posterior.mean(),scale=0.000000001))
                        layer.bias_posterior = tfd.Independent(tfd.Deterministic(loc=model.layers[i].bias_posterior.mean()))
                    except AttributeError:
                        continue

            def output_weights(model_out,fast_weights):
                j=0
                for i, layer in enumerate(model_out.layers):
                    print(i,layer)
                    print('j=',j)
                    try:
                        print(layer.kernel_posterior)
                        layer.kernel_posterior = tfd.Independent(tfd.Normal(loc=fast_weights[j],scale=tf.math.exp(fast_weights[j+1])))
                        layer.bias_posterior = tfd.Independent(tfd.Deterministic(loc=fast_weights[j+2]))
                        j+=3
                        print('tfp')

                    except AttributeError:
                        '''
                        layer.gamma = fast_weights[j]
                        layer.beta = fast_weights[j+1]
                        j+=2
                        print('norm')
                        '''
                        continue

            def neg_L(model, input, label):
                mean = model(tf.cast(input, tf.float32))
                if self.classification:
                    labels_distribution = tfd.Categorical(logits=mean)
                    label = tf.argmax(label, axis=1)
                else:
                    #labels_distribution = tfd.Normal(loc=mean ,scale= std) #???
                    labels_distribution = tfd.Normal(loc=mean ,scale= self.sigma)
                neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(label, tf.float32)))
                return neg_log_likelihood

            def set_prior(model_out,model):
                for i, layer in enumerate(model_out.layers):
                    try:
                        layer.kernel_prior = model.layers[i].kernel_posterior
                        layer.bias_prior = model.layers[i].bias_posterior
                    except AttributeError:
                        continue

            set_prior(weights_a,weights)
            set_prior(weights_b,weights)



            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb, task_lossesb_op = [], [], []
                if self.classification:
                    task_accuraciesb = []


                # initialize weights_a , fast_weights_a, weights_b, fast_weights_b
                fast_weights_a = weights.trainable_weights
                fast_weights_b = weights.trainable_weights
                output_weights(weights_a,fast_weights_a)
                output_weights(weights_b,fast_weights_b)
                # dumb_loss
                neg_log_likelihood_dumb_b = neg_L(weights, tf.concat([inputa,inputb],0) , tf.concat([labela,labelb],0))
                neg_log_likelihood_dumb_a = neg_L(weights, inputa , labela)
                # output and loss for initial state
                deter(weights_output,weights_a)
                task_output = weights_output(tf.cast(inputb, tf.float32))
                task_outputa = task_output
                task_lossa = self.loss_func(task_output, tf.cast(labelb, tf.float32))


                print('num_updates=',num_updates) #!!!!!!!!
                for j in range(num_updates):

                    # traditional val loss
                    neg_log_likelihood_cross = neg_L(weights_a,inputb,labelb)

                    # posterior a
                    neg_log_likelihood_a = neg_L(weights_a,inputa,labela)
                    kl = sum(weights_a.losses) / tf.cast(tf.size(inputa), tf.float32)  #???
    	            elbo_loss_a = neg_log_likelihood_a + kl
                    if j==0:
                        task_lossa_op = elbo_loss_a

                    grads_a = tf.gradients(elbo_loss_a, fast_weights_a)
                    if FLAGS.stop_grad:
                        grads_a = [tf.stop_gradient(grad) for grad in grads_a]

                    fast_weights_a = [(fast_weights_a[i] - self.update_lr*grads_a[i]) for i in range(len(grads_a))]
                    output_weights(weights_a,fast_weights_a)

                    deter(weights_output,weights_a)
                    task_output = weights_output(tf.cast(inputb, tf.float32))
                    task_outputbs.append(task_output)  #
                    task_lossesb.append(self.loss_func(task_output, tf.cast(labelb, tf.float32)))

                    # posterior b

                    neg_log_likelihood_b = neg_L(weights_b, tf.concat([inputa,inputb],0) , tf.concat([labela,labelb],0))
                    kl = sum(weights_b.losses) / tf.cast(tf.size(inputa)+tf.size(inputb), tf.float32)  #???
                    elbo_loss_b = neg_log_likelihood_b + kl
                    grads_b = tf.gradients(elbo_loss_b, fast_weights_b)
                    if FLAGS.stop_grad:
                        grads_b = [tf.stop_gradient(grad) for grad in grads_b]
                    fast_weights_b = [(fast_weights_b[i]  - self.update_lr*grads_b[i]) for i in range(len(grads_b))]
                    output_weights(weights_b,fast_weights_b)


                    # define the loss op
                    fw_a_stop = [tf.stop_gradient(weight) for weight in fast_weights_a]
                    output_weights(weights_a_stop,fw_a_stop)
                    fw_b_stop = [tf.stop_gradient(weight) for weight in fast_weights_b]
                    output_weights(weights_b_stop,fw_b_stop)

                    lossb_abq = []
                    lossb_ab_kl =[]
                    lossb_bq =[]
                    lossb_ab_xe=[]

                    for i, layer in enumerate(weights.layers):
                        try:
                            #q = layer.kernel_posterior
                            q = tfd.Independent(tfd.Normal(loc=layer.kernel_posterior.mean(),scale=layer.kernel_posterior.stddev()))
                            lossb_abq.append( - weights_a_stop.layers[i].kernel_posterior.cross_entropy(q) + weights_b_stop.layers[i].kernel_posterior.cross_entropy(q) )
                            lossb_ab_kl.append(weights_b_stop.layers[i].kernel_posterior.kl_divergence(weights_a.layers[i].kernel_posterior))
                            lossb_bq.append(weights_b_stop.layers[i].kernel_posterior.cross_entropy(q))
                            lossb_ab_xe.append(weights_b_stop.layers[i].kernel_posterior.cross_entropy(weights_a.layers[i].kernel_posterior))
                        except AttributeError:
                            continue


                    if FLAGS.meta_loss == 'abq':
                        meta_loss = sum(lossb_abq)
                    if FLAGS.meta_loss == 'ab_kl':
                        meta_loss = sum(lossb_ab_kl)
                    if FLAGS.meta_loss == 'ab_xe':
                        meta_loss = sum(lossb_ab_xe)
                    if FLAGS.meta_loss == 'bq':
                        meta_loss = sum(lossb_bq)
                    if FLAGS.meta_loss == 'b':
                        meta_loss = neg_log_likelihood_b
                    if FLAGS.meta_loss == 'b-a':
                        meta_loss = neg_log_likelihood_b - neg_log_likelihood_a
                    if FLAGS.meta_loss == 'b*a':
                        meta_loss = neg_log_likelihood_cross
                    if FLAGS.meta_loss == 'dumb_loss':
                        meta_loss = neg_log_likelihood_dumb_b
                    if FLAGS.meta_loss == 'dumb_b-a':
                        meta_loss = -(neg_log_likelihood_dumb_b - neg_log_likelihood_dumb_a)
                    task_lossesb_op.append(meta_loss)
                # end for
                #self.outb_last=task_outputb    #!!!!!!!!!
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, task_lossa_op, task_lossesb_op ]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output



            # meta-train
#            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
#                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates , tf.float32, [tf.float32]*num_updates]

            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            #result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            outputas, outputbs, lossesa, lossesb, lossesa_op, lossesb_op, accuraciesa, accuraciesb = [None]*N_task,[None]*N_task,[None]*N_task,[None]*N_task,[None]*N_task,[None]*N_task,[None]*N_task, [None]*N_task
            for i in range(N_task):
                elems=(self.inputa[i], self.inputb[i], self.labela[i], self.labelb[i])
                if self.classification:
                    outputas[i], outputbs[i], lossesa[i], lossesb[i], lossesa_op[i], lossesb_op[i], accuraciesa[i], accuraciesb[i] = task_metalearn(elems)
                else:
                    outputas[i], outputbs[i], lossesa[i], lossesb[i], lossesa_op[i], lossesb_op[i]  = task_metalearn(elems)

            self.outputas = outputas
            self.outputbs = outputbs
            self.lossesa = lossesa
            self.lossesb = lossesb
            #result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb, task_number), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)


        ## Performance & Optimization
        print('num_updates=',num_updates,'\n')
        #print('outshape=',lossesa.shape)
        lossesb = tf.transpose(lossesb)   # get the correct form of lossesb
        lossesb_op = tf.transpose(lossesb_op)
        if self.classification:
            accuraciesb = tf.transpose(accuraciesb)
            self.accuraciesb = accuraciesb


        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            #lossesb = tf.transpose(lossesb)   # get the correct form of lossesb
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            #self.total_losses2 = total_losses2 = [lossesb[j] for j in range(num_updates)]
            #self.total_losses2 = total_losses2 = lossesb[0]
            self.total_loss1_op = total_loss1_op = tf.reduce_sum(lossesa_op) / tf.to_float(FLAGS.meta_batch_size)
            #lossesb_op_trans = tf.transpose(lossesb_op)
            self.total_losses2_op = total_losses2_op = [tf.reduce_sum(lossesb_op[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            #self.total_losses2_op = total_losses2_op = [lossesb_op[j] for j in range(num_updates)]
            #self.total_losses2_op = total_losses2_op = lossesb_op[0]
            # after the map_fn
            #self.outputas, self.outputbs = outputas, outputbs

            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1_op)


 ######  ##### optimize meta loss func
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2_op[FLAGS.num_updates-1])
                if FLAGS.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            #self.metaval_total_losses2 = total_losses2 = lossesb[0]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

#########################

        ## Summaries

        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            print('total_losses2[j]=',total_losses2[j])
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])
