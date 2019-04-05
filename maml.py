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
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
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
            tf.keras.layers.Reshape((-1,channels,self.img_size, self.img_size)),
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
        else:
            #print('its not none fuck')
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb'] 
            self.inputa_init = self.inputa
        

        self.sigma = FLAGS.sigma    
        self.num_repeat = FLAGS.num_repeat
        '''
        self.inputa = input_tensors['inputa'] 
        self.inputb = input_tensors['inputb'] #  !!!!!debug
        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb'] #  !!!!!debug

        
        #self.inputa_test = self.inputa 
        if FLAGS.train:
            self.inputa_test = self.inputa
        else: 
            self.inputa_test = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        
        '''
        #print(self.inputa.shape)
        #print(self.labela.shape)
        #print(self.inputa[:,1,:])
        #print(self.inputa[1])
        
        #N_task = self.inputa.shape[0]
        self.task_number=0
        self.weights_a, self.weights_b, self.weights_output, self.weights_test = [],[],[],[]
        self.weights_a_stop, self.weights_b_stop = [],[]
        #task_number = np.array(range(N_task))
	#print('N_task=',N_task)
        N_task=FLAGS.meta_batch_size
        print('N_task=',N_task) 
        for i in range(N_task):
            weights_a = self.construct_weights()
            weights_a((self.inputa_init[0]).astype('float32'))
            weights_b = self.construct_weights()
            weights_b((self.inputa_init[0]).astype('float32'))
            weights_a_stop = self.construct_weights()
            weights_a_stop((self.inputa_init[0]).astype('float32'))
            weights_b_stop = self.construct_weights()
            weights_b_stop((self.inputa_init[0]).astype('float32'))
            weights_output = self.construct_weights()
            weights_output((self.inputa_init[0]).astype('float32'))
            self.weights_a.append(weights_a)
            self.weights_b.append(weights_b)
            self.weights_a_stop.append(weights_a_stop)
            self.weights_b_stop.append(weights_b_stop)
            self.weights_output.append(weights_output)

        
	    weights_test = self.construct_weights()
            weights_test((self.inputa_init[0]).astype('float32'))
            self.weights_test.append(weights_test)

       
	 #   print(self.labela)
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):   
                training_scope.reuse_variables()
                weights = self.weights
                '''
                weights_a = self.weights_a
                weights_b = self.weights_b
                weights_output = self.weights_output
		'''
            else:
                # Define the weights /  weights stands for the model nueral_net!!!!!!!
                # run models with array input to initialize models
                #random.seed(7)
                if FLAGS.datasource == 'sinusoid':
                    self.weights = weights = self.construct_weights()
                    weights((self.inputa_init[0]).astype('float32'))
                    self.weights_cp = weights_cp = self.construct_weights()
                    weights_cp((self.inputa_init[0]).astype('float32'))
                    '''
                    self.weights_a = weights_a = self.construct_weights()
                    weights_a((self.inputa[0]).astype('float32'))
                    self.weights_b = weights_b = self.construct_weights()
                    weights_b((self.inputa[0]).astype('float32'))
                    self.weights_output = weights_output = self.construct_weights()
                    weights_output((self.inputa[0]).astype('float32'))
                    self.weights_test = weights_test = self.construct_weights()
                    weights_test((self.inputa[0]).astype('float32'))
                    '''
                else:
                    self.weights = weights = self.construct_weights()
                    #weights(self.inputa[0])
                    #self.weights_a = weights_a = self.construct_weights()
                    #weights_a(self.inputa[0])
                    #self.weights_b = weights_b = self.construct_weights()
                    #weights_b(self.inputa[0])


            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            #num_updates = max(self.test_num_updates, FLAGS.num_updates)
            num_updates = FLAGS.num_updates
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates   

            self.weights_cp.layers[0].kernel_posterior=self.weights.layers[0].kernel_posterior       
            
            #self.weights_test[0].layers[0].kernel_posterior = tfd.Independent(tfd.Normal(loc=weights.layers[0].kernel_posterior.mean(),scale=0.000000001))

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb, task_lossesb_op = [], [], []
                weights_a = self.weights_a[self.task_number]
                weights_b = self.weights_b[self.task_number]
                weights_a_stop = self.weights_a_stop[self.task_number]
                weights_b_stop = self.weights_b_stop[self.task_number]
                weights_output = self.weights_output[self.task_number]
                weights_test=self.weights_test[self.task_number]
                self.task_number = self.task_number + 1


                if self.classification:
                    task_accuraciesb = []
                # first gradient step  

                def predict(NN,inputs):
                    logits=[]
                    #print('num_repeat=',self.num_repeat)
                    for i in range(self.num_repeat):
                        logits.append(NN(inputs))
                    mean = tf.reduce_mean(logits,0)
                    std = reduce_std(logits,0)
                    #print(mean,std)
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
                            layer.gamma = fast_weights[j]
                            layer.beta = fast_weights[j+1]
                            j+=2
                            print('norm')
                            continue
               


                # dumb_loss
                mean_db , std_db = predict(weights,tf.concat([inputa,inputb],0))
                #labels_distribution = tfd.Normal(loc=mean ,scale= std) #???
                labels_distribution_db = tfd.Normal(loc=mean_db ,scale= self.sigma)
                neg_log_likelihood_dumb = -tf.reduce_mean(labels_distribution_db.log_prob(tf.cast(tf.concat([labela,labelb],0), tf.float32)))

                #

 
                if FLAGS.datasource == 'omniglot': 
                    channels = self.channels
                    #inputa = tf.reshape(inputa, [-1, self.img_size, self.img_size, channels])
                    #inputb = tf.reshape(inputb, [-1, self.img_size, self.img_size, channels])
                if not self.classification:
                    mean , std = predict(weights,inputa)
                    
                else:
                    #print(tf.cast(inputa, tf.float32))
                    mean = weights(tf.cast(inputa, tf.float32))

                #deter(weights_output,weights)
                print('task_number=',self.task_number)
                #self.weights_test[self.task_number-1].layers[0].kernel_posterior = tfd.Independent(tfd.Normal(loc=self.weights_cp.layers[0].kernel_posterior.mean(),scale=0.000000001))
                #self.weights_test[self.task_number-1].layers[0].kernel_posterior = tfd.Independent(tfd.Normal(loc=self.weights.layers[0].kernel_posterior.mean(),scale=0.000000001))               
                '''
                for i, layer in enumerate(self.weights_test[self.task_number-1].layers):
                    try:
                        self.weights_test[self.task_number-1].layers[i].kernel_posterior = tfd.Independent(tfd.Normal(loc=self.weights.layers[i].kernel_posterior.mean(),scale=0.00000000000000000000001))
                        print('layer',i)
                    except AttributeError:
                        continue 
                '''



                deter(self.weights_test[self.task_number-1],self.weights)
                #self.inputa_check = self.inputa_test
                task_outputa = self.weights_test[self.task_number-1](tf.cast(self.inputa[0], tf.float32))  #!!! maybe wrong
                self.task_outputa = task_outputa #debug!!!!
                self.task_outputa_test = self.weights(tf.cast(self.inputa[0], tf.float32))
                #init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		#sess=tf.InteractiveSession()
                #tf.global_variables_initializer().run()
    		    #sess.run(init_op)
                #print(sess.run(task_outputa))
                task_lossa = self.loss_func(task_outputa, tf.cast(labela, tf.float32))
                self.task_lossa = task_lossa # debug!!!!
                
                if self.classification:
                    labels_distribution = tfd.Categorical(logits=mean)
                else:
                    #labels_distribution = tfd.Normal(loc=mean ,scale= std) #???     
                    labels_distribution = tfd.Normal(loc=mean ,scale= self.sigma)
                neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(labela, tf.float32)))
                kl = sum(weights.losses) / tf.cast(tf.size(inputa), tf.float32)  #???               
                elbo_loss_a = neg_log_likelihood + kl                
                task_lossa_op = elbo_loss_a
                grads_a = tf.gradients(elbo_loss_a, weights.trainable_weights) 
                self.grads_1 = grads_a #!!!!!!
                
                 
                '''
                    if FLAGS.stop_grad:
                        grads_a = [tf.stop_gradient(grad) for grad in grads] 
                    '''    
                # initialize                
                #weights_a(tf.cast(inputa, tf.float32)) 
                #weights_b(tf.cast(inputa, tf.float32))
                print('lr=',self.update_lr) 
                #self.update_lr = 0.1
               
                fast_weights_a = [(weights.trainable_weights[i] - self.update_lr*grads_a[i]) for i in range(len(grads_a))]
                #print(weights.trainable_variables)
                '''
                for i in range(len(true_weights_a)):
                    #tf.assign(weights_a.trainable_variables[i] ,true_weights_a[i] )
                    weights_a.trainable_weights[i] = true_weights_a[i]
                '''
                output_weights(weights_a,fast_weights_a)
 
                #self.watw = weights_a.trainable_weights
                #weights_a.trainable_weights = true_weights_a
                #self.true_weights_a = true_weights_a # !!!!!!
                 
                #!!!!! debug
               


                # posterior b
                if not self.classification:
                    mean , std = predict(weights_a,inputb)
                else: 
                    mean = weights_a(tf.cast(inputb, tf.float32))
                
                self.wa = weights_a  #!!!!!!!!
                deter(weights_output,weights_a)
                self.wo = weights_output   #!!!!!!
                task_outputb = weights_output(tf.cast(inputb, tf.float32)) 
                self.outb = task_outputb #!!!!!
                task_outputbs.append(task_outputb)  #!!!maybe wrong
                task_lossesb.append(self.loss_func(task_outputb, tf.cast(labelb, tf.float32))) 
                self.lb = self.loss_func(task_outputb, tf.cast(labelb, tf.float32))   #!!!!!!
                if self.classification:
                    labels_distribution = tfd.Categorical(logits=mean)
                else:
                    #labels_distribution = tfd.Normal(loc=mean ,scale= std) #???
                    labels_distribution = tfd.Normal(loc=mean ,scale= self.sigma)
                neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(labelb, tf.float32)))               
                kl = sum(weights_a.losses) / tf.cast(tf.size(inputb), tf.float32)  #???
                elbo_loss_b = neg_log_likelihood + kl 
                #grads_b = tf.gradients(elbo_loss_b, weights_a.trainable_weights)        
                grads_b = tf.gradients(elbo_loss_b, fast_weights_a) 
                fast_weights_b = [(fast_weights_a[i]  - self.update_lr*grads_b[i]) for i in range(len(grads_b))]     #!!!!!          
                '''
                for i in range(len(true_weights_b)):
                    #tf.assign(weights_b.trainable_variables[i] ,true_weights_b[i] )
                    weights_b.trainable_weights[i] = true_weights_b[i]
                '''
                output_weights(weights_b,fast_weights_b)

                
                '''
                if (num_updates==1):
                    for i in range(len(weights_a.trainable_variables)):
                        #tf.assign(weights_a.trainable_variables[i] ,tf.stop_gradient(weights_a.trainable_variables[i]) )
                        weights_a.trainable_variables[i] = tf.stop_gradient(weights_a.trainable_variables[i])
                    for i in range(len(weights_a.trainable_variables)):
                        #tf.assign(weights_a.trainable_variables[i] ,tf.stop_gradient(weights_a.trainable_variables[i]) )
                        weights_b.trainable_variables[i] = tf.stop_gradient(weights_b.trainable_variables[i])
                '''

                #weights_a_s = [tf.stop_gradient(weight) for weight in weights_a]
                #weights_b_s = [tf.stop_gradient(weight) for weight in weights_b]
                fw_a_stop = [tf.stop_gradient(weight) for weight in fast_weights_a]
                output_weights(weights_a_stop,fw_a_stop)
                fw_b_stop = [tf.stop_gradient(weight) for weight in fast_weights_b]
                output_weights(weights_b_stop,fw_b_stop)

                lossb_abq = []
                lossb_ab_kl =[]
                lossb_bq =[]

                for i, layer in enumerate(weights.layers):
                    try:
                        #q = layer.kernel_posterior
                        q = tfd.Independent(tfd.Normal(loc=layer.kernel_posterior.mean(),scale=layer.kernel_posterior.stddev()))
                        lossb_abq.append( - weights_a_stop.layers[i].kernel_posterior.cross_entropy(q) + weights_b_stop.layers[i].kernel_posterior.cross_entropy(q) )
                        lossb_ab_kl.append(weights_b_stop.layers[i].kernel_posterior.kl_divergence(weights_a.layers[i].kernel_posterior) )
                        lossb_bq.append(weights_b_stop.layers[i].kernel_posterior.cross_entropy(q))
                    except AttributeError:
                        continue

                if FLAGS.meta_loss == 'chaser_loss':
                    meta_loss = sum(lossb_abq)
                if FLAGS.meta_loss == 'chaser_loss_kl':
                    meta_loss = sum(lossb_ab_kl)
                if FLAGS.meta_loss == 'mix_effect':
                    meta_loss = sum(lossb_bq)
                if FLAGS.meta_loss == 'val_loss':
                    meta_loss = neg_log_likelihood
            
                if FLAGS.meta_loss == 'traditional_val_loss':   # this one is true
                    meta_loss = neg_log_likelihood

                if FLAGS.meta_loss == 'dumb_loss':    
                    meta_loss = neg_log_likelihood_dumb   

                task_lossesb_op.append(meta_loss)
                
                print('num_updates=',num_updates) #!!!!!!!!
                # the rest gradient steps
                for j in range(num_updates-1): 
                    # posterior a 
                    
                    #logits = weights_a(tf.cast(inputa, tf.float32))
                    if not self.classification:
                        mean , std = predict(weights_a,inputa)
                    else: 
                        mean = weights_a(tf.cast(inputa, tf.float32))

                    if self.classification:
                        labels_distribution = tfd.Categorical(logits=mean)
                    else:
                        #labels_distribution = tfd.Normal(loc=mean ,scale= std) #???    
                        labels_distribution = tfd.Normal(loc=mean ,scale= self.sigma)       
                    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(labela, tf.float32)))
                    # traditional val loss
                    mean_tvl , std_tvl = predict(weights_a,inputb)
                    labels_distribution_tvl = tfd.Normal(loc=mean_tvl ,scale= self.sigma) 
                    neg_log_likelihood_tvl = -tf.reduce_mean(labels_distribution_tvl.log_prob(tf.cast(labelb, tf.float32)))
                    #

                    kl = sum(weights_a.losses) / tf.cast(tf.size(inputa), tf.float32)  #???               
    	            elbo_loss_a = neg_log_likelihood + kl           
                    #grads_a = tf.gradients(elbo_loss_a, weights_a.trainable_weights)
                    grads_a = tf.gradients(elbo_loss_a, fast_weights_a)
                    #if FLAGS.stop_grad:
                    #    grads_a = [tf.stop_gradient(grad) for grad in grads] 
                                       
                    fast_weights_a = [(fast_weights_a[i] - self.update_lr*grads_a[i]) for i in range(len(grads_a))]
                    '''
                    for i in range(len(true_weights_a)):
                        #tf.assign(weights_a.trainable_variables[i] ,true_weights_a[i] )
                        weights_a.trainable_weights[i]=true_weights_a[i]
     #               print('set weight successfully')
                    '''
                    output_weights(weights_a,fast_weights_a)

                    #output = tf.argmax(weights_a(tf.cast(inputb, tf.float32)), axis=1)
                    if not self.classification:
                        mean , std = predict(weights_a,inputb)
                    else: 
                        mean = weights_a(tf.cast(inputb, tf.float32))  
                    deter(weights_output,weights_a)
                    task_outputb = weights_output(tf.cast(inputb, tf.float32)) 
                    task_outputbs.append(task_outputb)  #!!!maybe wrong
                    #task_outputbs.append(mean) #!!!maybe wrong
                    task_lossesb.append(self.loss_func(task_outputb, tf.cast(labelb, tf.float32)))
                    
                    # posterior b
                    #logits = weights_b(tf.cast(tf.concat([inputa,inputb],0), tf.float32))
                    #mean , std = predict(weights_b,tf.concat([inputa,inputb],0))
                    if self.classification:
                        #mean , std = predict(weights_b,inputa)
                        mean = weights_b(tf.cast(inputa, tf.float32))  
                        labels_distribution_a = tfd.Categorical(logits=mean)
                        neg_log_likelihood_a = -tf.reduce_mean(labels_distribution_a.log_prob(tf.cast(labela, tf.float32)))               
                        #mean , std = predict(weights_b,inputb)
                        mean = weights_b(tf.cast(inputb, tf.float32))  
                        labels_distribution_b = tfd.Categorical(logits=mean)
                        neg_log_likelihood_b = -tf.reduce_mean(labels_distribution_b.log_prob(tf.cast(labelb, tf.float32)))               
                        neg_log_likelihood = neg_log_likelihood_a + neg_log_likelihood_b

                    else:
                        mean , std = predict(weights_b,tf.concat([inputa,inputb],0))
                        #labels_distribution = tfd.Normal(loc=mean ,scale= std) #???
                        labels_distribution = tfd.Normal(loc=mean ,scale= self.sigma)
                        neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(tf.cast(tf.concat([labela,labelb],0), tf.float32)))       

                    kl = sum(weights_b.losses) / tf.cast(tf.size(inputa)+tf.size(inputb), tf.float32)  #???
                    elbo_loss_b = neg_log_likelihood + kl
                    grads_b = tf.gradients(elbo_loss_b, fast_weights_b)                 
                    fast_weights_b = [(fast_weights_b[i]  - self.update_lr*grads_b[i]) for i in range(len(grads_b))]               
                    '''
                    for i in range(len(true_weights_b)):
                        #tf.assign(weights_b.trainable_variables[i] ,true_weights_b[i] )
                        weights_b.trainable_weights[i]=true_weights_b[i]
                    '''
                    output_weights(weights_b,fast_weights_b)

                    
#                    weights_a = [tf.stop_gradient(weight) for weight in weights_a]
#                    weights_b = [tf.stop_gradient(weight) for weight in weights_b]
                    '''
                    if j == (num_updates-2):
                        for i in range(len(weights_a.trainable_variables)):
                            #tf.assign(weights_a.trainable_variables[i] ,tf.stop_gradient(weights_a.trainable_variables[i]) ) 
                            weights_a.trainable_variables[i] = tf.stop_gradient(weights_a.trainable_variables[i])
                        for i in range(len(weights_a.trainable_variables)):
                            #tf.assign(weights_a.trainable_variables[i] ,tf.stop_gradient(weights_a.trainable_variables[i]) )
                            weights_b.trainable_variables[i] = tf.stop_gradient(weights_b.trainable_variables[i])
                    '''
                    fw_a_stop = [tf.stop_gradient(weight) for weight in fast_weights_a]
                    output_weights(weights_a_stop,fw_a_stop)
                    fw_b_stop = [tf.stop_gradient(weight) for weight in fast_weights_b]
                    output_weights(weights_b_stop,fw_b_stop)

                    lossb_abq = []
                    lossb_ab_kl =[]
                    lossb_bq =[]

                    for i, layer in enumerate(weights.layers):
                        try:
                            #q = layer.kernel_posterior
                            q = tfd.Independent(tfd.Normal(loc=layer.kernel_posterior.mean(),scale=layer.kernel_posterior.stddev()))
                            lossb_abq.append( - weights_a_stop.layers[i].kernel_posterior.cross_entropy(q) + weights_b_stop.layers[i].kernel_posterior.cross_entropy(q) )
                            lossb_ab_kl.append(weights_b_stop.layers[i].kernel_posterior.kl_divergence(weights_a.layers[i].kernel_posterior) )
                            lossb_bq.append(weights_b_stop.layers[i].kernel_posterior.cross_entropy(q))
                        except AttributeError:
                            continue

                    if FLAGS.meta_loss == 'chaser_loss':
                        meta_loss = sum(lossb_abq)
                    if FLAGS.meta_loss == 'chaser_loss_kl':
                        meta_loss = sum(lossb_ab_kl)
                    if FLAGS.meta_loss == 'mix_effect':
                        meta_loss = sum(lossb_bq)
                    if FLAGS.meta_loss == 'val_loss':
                        meta_loss = neg_log_likelihood
                    if FLAGS.meta_loss == 'traditional_val_loss':
                        meta_loss = neg_log_likelihood_tvl
                    if FLAGS.meta_loss == 'dumb_loss':
                        meta_loss = neg_log_likelihood_dumb


                    task_lossesb_op.append(meta_loss)
                
                self.outb_last=task_outputb    #!!!!!!!!!
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
            #out_dtype = [ tf.float32, tf.float32 ] 
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            #result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            #result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype)

            outputas, outputbs, lossesa, lossesb, lossesa_op, lossesb_op = [None]*N_task,[None]*N_task,[None]*N_task,[None]*N_task,[None]*N_task,[None]*N_task
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
            #print(lossesb)
            ''' 
            lossesb = np.transpose(np.array(lossesb))
            lossesb_op = np.transpose(np.array(lossesb_op)) 
            outputbs = np.transpose(np.array(outputbs))
           
            lossesb = lossesb[0]
            lossesb_op = lossesb_op[0]
            outputbs = outputbs[0]
            '''
            #lossesb = [list(i) for i in zip(*lossesb)]
            
            #print(lossesb[0])    
            
            '''
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb, task_number), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
          
            if self.classification:
                outputas, outputbs, lossesa, lossesb, lossesa_op, lossesb_op, accuraciesa, accuraciesb = result              
            else:
                outputas, outputbs, lossesa, lossesb, lossesa_op, lossesb_op  = result
            '''

        ## Performance & Optimization
        print('num_updates=',num_updates,'\n')
        #print('outshape=',lossesa.shape)
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            lossesb_trans = tf.transpose(lossesb)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb_trans[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            #self.total_losses2 = total_losses2 = [lossesb[j] for j in range(num_updates)]
            #self.total_losses2 = total_losses2 = lossesb[0]
            self.total_loss1_op = total_loss1_op = tf.reduce_sum(lossesa_op) / tf.to_float(FLAGS.meta_batch_size)
            lossesb_op_trans = tf.transpose(lossesb_op)
            self.total_losses2_op = total_losses2_op = [tf.reduce_sum(lossesb_op_trans[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
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
            #self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.metaval_total_losses2 = total_losses2 = lossesb[0]
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
        



