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
#from tensorflow_probability.python import distributions as tfd
'''
tfe = tf.contrib.eager
try:
    tfe.enable_eager_execution()
except ValueError:
    pass
'''
FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
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
        model = tf.keras.Sequential([
            tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=k, padding="SAME", activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
            tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=5, padding="SAME", activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
            tfp.layers.Convolution2DFlipout(self.dim_hidden, kernel_size=5, padding="SAME",activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
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
        model.add(tfp.layers.DenseFlipout(self.dim_output))
        
     
        return model


    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        '''
        if input_tensors is None:
            print('fuck its none')
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            print('its not none fuck')
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']
        '''
        self.sigma = 0.5    # change to flag later
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb']

	 #   print(self.labela)
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):   
                training_scope.reuse_variables()
                weights = self.weights
                weights_a = self.weights_a
                weights_b = self.weights_b

            else:
                # Define the weights /  weights stands for the model nueral_net!!!!!!!
                # run models with array input to initialize models
                self.weights = weights = self.construct_weights()
                weights((self.inputa[0]).astype('float32'))
                self.weights_a = weights_a = self.construct_weights()
                weights_a((self.inputa[0]).astype('float32'))
                self.weights_b = weights_b = self.construct_weights()
                weights_b((self.inputa[0]).astype('float32'))

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates            
            

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                if self.classification:
                    task_accuraciesb = []
                # first gradient step
                logits = weights(tf.cast(inputa, tf.float32))
                task_outputa = logits  #!!! maybe wrong
                if self.classification:
                    labels_distribution = tfd.Categorical(logits=logits)
                else:
                    labels_distribution = tfd.Normal(loc=logits ,scale= self.sigma) #???           
                neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(labela, tf.float32)))
                kl = sum(weights.losses) / tf.cast(tf.size(inputa), tf.float32)  #???               
                elbo_loss_a = neg_log_likelihood + kl                
                task_lossa = elbo_loss_a
                grads_a = tf.gradients(elbo_loss_a, weights.trainable_weights) 
                '''
                    if FLAGS.stop_grad:
                        grads_a = [tf.stop_gradient(grad) for grad in grads] 
                    '''                    
                true_weights_a = [(weights.trainable_weights[i] - self.update_lr*grads_a[i]) for i in range(len(grads_a))]
                for i in range(len(true_weights_a)):
                    tf.assign(weights_a.trainable_variables[i] ,true_weights_a[i] )

                # posterior b
                logits = weights_a(tf.cast(inputb, tf.float32))
                task_outputbs.append(logits)  #!!!maybe wrong
                if self.classification:
                    labels_distribution = tfd.Categorical(logits=logits)
                else:
                    labels_distribution = tfd.Normal(loc=logits ,scale= self.sigma) #???
                neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(labelb, tf.float32)))               
                kl = sum(weights_a.losses) / tf.cast(tf.size(inputb), tf.float32)  #???
                elbo_loss_b = neg_log_likelihood + kl
                grads_b = tf.gradients(elbo_loss_b, weights_a.trainable_weights)                 
                true_weights_b = [(weights_a.trainable_weights[i]  - self.update_lr*grads_b[i]) for i in range(len(grads_b))]               
                for i in range(len(true_weights_b)):
                    tf.assign(weights_b.trainable_variables[i] ,true_weights_b[i] )

                lossb = []
                for i, layer in enumerate(weights.layers):
                    try:
                        q = layer.kernel_posterior
                        lossb.append( weights_a.layers[i].kernel_posterior.cross_entropy(q) - weights_b.layers[i].kernel_posterior.cross_entropy(q) )
                    except AttributeError:
                        continue
                task_lossesb.append(sum(lossb))

                # the rest gradient steps
                for j in range(num_updates-1): 
                    # posterior a 
             
                    logits = weights_a(tf.cast(inputa, tf.float32))
                    if self.classification:
                        labels_distribution = tfd.Categorical(logits=logits)
                    else:
                        labels_distribution = tfd.Normal(loc=logits ,scale= self.sigma) #???           
                    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(labela, tf.float32)))
                    kl = sum(weights_a.losses) / tf.cast(tf.size(inputa), tf.float32)  #???               
    	            elbo_loss_a = neg_log_likelihood + kl           
                    grads_a = tf.gradients(elbo_loss_a, weights_a.trainable_weights) 
		    
                    #if FLAGS.stop_grad:
                    #    grads_a = [tf.stop_gradient(grad) for grad in grads] 
                                       
                    true_weights_a = [(weights_a.trainable_weights[i] - self.update_lr*grads_a[i]) for i in range(len(grads_a))]
                    for i in range(len(true_weights_a)):
                        tf.assign(weights_a.trainable_variables[i] ,true_weights_a[i] )
     #               print('set weight successfully')

                    # posterior b
                    logits = weights_b(tf.cast(tf.concat([inputa,inputb],0), tf.float32))
                    task_outputbs.append(logits) #!!!maybe wrong
                    if self.classification:
                        labels_distribution = tfd.Categorical(logits=logits)
                    else:
                        labels_distribution = tfd.Normal(loc=logits ,scale= self.sigma) #???
                    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(tf.concat([labela,labelb],0), tf.float32)))               
                    kl = sum(weights_b.losses) / tf.cast(tf.size(inputa)+tf.size(inputb), tf.float32)  #???
                    elbo_loss_b = neg_log_likelihood + kl
                    grads_b = tf.gradients(elbo_loss_b, weights_b.trainable_weights)                 
                    true_weights_b = [(weights_b.trainable_weights[i]  - self.update_lr*grads_b[i]) for i in range(len(grads_b))]               
                    for i in range(len(true_weights_b)):
                        tf.assign(weights_b.trainable_variables[i] ,true_weights_b[i] )

                    lossb = []
                    for i, layer in enumerate(weights.layers):
                        try:
                            q = layer.kernel_posterior
                            lossb.append( weights_a.layers[i].kernel_posterior.cross_entropy(q) - weights_b.layers[i].kernel_posterior.cross_entropy(q) )
                        except AttributeError:
                            continue
                    task_lossesb.append(sum(lossb))
                
              
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]  
                
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

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates ]
            #out_dtype = [ tf.float32, tf.float32 ] 
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
          
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result              
            else:
                outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
                

 ######  ##### optimize meta loss func                
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                if FLAGS.datasource == 'miniimagenet': 
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
          
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

#########################
        
        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])           
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])
        



