"""
Usage Instructions:


!!!!!!!!!!!!!!!!!!  FLAGS.stop_grad = TRUE !!!!!!!!!!!!!!!!!!!!
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""

import csv
import numpy as np
import pickle
import random
import os
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

import tensorflow_probability as tfp
tfd = tfp.distributions

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_bool('hard_sin', False, 'if true, generate harder sinusoid data')
flags.DEFINE_float('noise_factor', 0.01, 'noise_factor')
flags.DEFINE_integer('train_total_num_tasks', -1, 'total number of tasks for training with finite dataset')
flags.DEFINE_integer('test_total_num_tasks', -1, 'total number of tasks for evaluation')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_float('sigma', 0.1, 'scale of label distribution')
flags.DEFINE_integer('num_repeat', 1, 'number of repeated runnings for each prediction')
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('switchtrain_iterations', -1, 'number of switch-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 10, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_float('prior_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 10, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_string('meta_loss', 'b*a', 'type of the meta loss function.')
flags.DEFINE_bool('one_sample', False, 'use the same sample for all training iterations or not')
flags.DEFINE_bool('setseed', False, 'use the same seed in one loop')
flags.DEFINE_float('dev_weight', 1.0, 'weight of dev different in meta loss func')
flags.DEFINE_integer('num_ll_samples', 1, 'number of samples in calculating neg_log_likelihood.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad',False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_bool('determ',False, 'if True, use the original deterministic NN (for DEBUG)')
flags.DEFINE_bool('redim',False, 'if True, reinterpreted_batch_ndims set to real dims')
flags.DEFINE_bool('inverse',False, 'if True, inverse the maml loss function')
flags.DEFINE_bool('separate_prior',False, 'if True, separate the training of prior and initial point')
flags.DEFINE_bool('no_prior',False, 'if True, set KL=0 in inner-update steps')
flags.DEFINE_bool('meta_elbo',False, 'if True, use elbo_loss in meta_update steps')
flags.DEFINE_bool('task_average',False, 'if True, save task average model weights for testing')
flags.DEFINE_bool('task_b',False, 'if True, use the last model_b weights for testing')
flags.DEFINE_bool('weightsb',False, 'if True, use weights <= weights_b as meta_update')
flags.DEFINE_bool('inputa_only',False, 'if True, use inputa and labela to train weights_b')


## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train.')
flags.DEFINE_bool('test', True, 'True to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot
flags.DEFINE_integer('test_num_updates', -1, 'number of inner gradient updates during testing.')
flags.DEFINE_bool('debug',False, 'if True, only very few samples will be generated in data (for DEBUG)')
flags.DEFINE_string('alias', '', 'it is used to distinguish different models in saving paths')
flags.DEFINE_string('model_file', None, 'model file for continual training')
flags.DEFINE_bool('construct_only',False, 'if True, only construct models and skip training or testing')
flags.DEFINE_bool('print_grads_details',False, 'if True, print details like gradients, model parameters')
flags.DEFINE_bool('test_all',False, 'if True, test on saved models of all training iterations')


from subprocess import check_output
def nvidia_smi(options=['-q','-d','MEMORY']):
    return check_output(['nvidia-smi'] + options)

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000   #!!!!!!! 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        if FLAGS.datasource == 'miniimagenet':
            PRINT_INTERVAL = 1
        else:
            PRINT_INTERVAL = 100   #!!!! 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    # one sample pretrain for sinusoid
    if 'generate' in dir(data_generator):
        batch_x, batch_y, amp, phase = data_generator.generate()
        inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
        labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
        inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
        labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
    #
    average_itr = 0 #    task-average algorithm
    #np.random.seed(1)
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):

        print(nvidia_smi())
        if not FLAGS.one_sample:
            feed_dict = {}

            if 'generate' in dir(data_generator):
                if (FLAGS.train_total_num_tasks > 0) and (itr % FLAGS.train_total_num_tasks) == 0:
                    np.random.seed(1)

                batch_x, batch_y, amp, phase = data_generator.generate()

                if FLAGS.baseline == 'oracle':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    for i in range(FLAGS.meta_batch_size):
                        batch_x[i, :, 1] = amp[i]
                        batch_x[i, :, 2] = phase[i]

                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                #print('inputa=',inputa)
                #print('inputa.shape=',inputa.shape,inputa)
                #print('inputb.shape=',inputb.shape,inputb)
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        # bmaml  -> abq
        if itr == FLAGS.switchtrain_iterations:
            ws = sess.run(model.weights.trainable_weights,feed_dict)
            print('weights=',ws)
            for i in range(len(ws)):
                sess.run( tf.assign(model.prior_weights.trainable_weights[i],ws[i]) )
            print('prior_weights=',sess.run(model.prior_weights.trainable_weights,feed_dict))
            FLAGS.separate_prior = True
            FLAGS.meta_loss = 'abq'

        if itr < FLAGS.pretrain_iterations:
	        input_tensors = [model.pretrain_op]
        else:
##### metatrain_op

            if FLAGS.weightsb:
                wb = sess.run(model.fast_weights_b,feed_dict)
                if (itr!=0) and itr % PRINT_INTERVAL == 0:
                    print('fast_weights_b=',wb)
                input_tensors = [tf.assign(model.weights.trainable_weights[i],wb[i]) for i in range(len(wb))]
            else:
                if FLAGS.separate_prior:
                    input_tensors = [model.metatrain_op, model.priortrain_op]
                else:
                    input_tensors = [model.metatrain_op]


        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])


        result = sess.run(input_tensors, feed_dict)
        # task-average  algorithm
        if FLAGS.task_average:
            if itr > FLAGS.metatrain_iterations/2:
                w = sess.run(model.fast_weights_b,feed_dict)
                if (itr!=0) and itr % PRINT_INTERVAL == 0:
                    print('fast_weights_b=',w)
                if average_itr==0:
                    task_weights_sum = w
                else:
                    for i in range(len(task_weights_sum)):
                        task_weights_sum[i] += w[i]

                average_itr += 1

        #print('result_debug=',result_debug)
        #print('result=',result,'\n')
        #print('result=',result[0],'\n')
        #print('result=',result[1],'\n')
        #print('result=',result[2],'\n')
        #print('result=',result[3],'\n')
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[-3], itr)
                '''
                if FLAGS.separate_prior:
                    train_writer.add_summary(result[2], itr)
                else:
                    train_writer.add_summary(result[1], itr)
                '''

            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)




            # # DEBUG: print layer kernel stddev

            if FLAGS.datasource == 'sinusoid':
                if FLAGS.print_grads_details:
                    for i, layer in enumerate(model.weights.layers):
                        try:
                            print('layer_mean',i,sess.run(layer.kernel_posterior.mean(), feed_dict))
                            print('layer_stddev',i,sess.run(layer.kernel_posterior.stddev(), feed_dict))
                        except AttributeError:
                            continue


                # print gradients
                #print_gvs = [ ('None',var) if grad is None else (grad,var) for grad, var in model.gvs]

                    for grad, var in model.gvs:
                        if grad is not None:
                            print(sess.run(grad, feed_dict), var, sess.run(var, feed_dict))
                    # print weights_b
                    for i, layer in enumerate(model.weights_b.layers):
                        try:
                            print('weights_b_layer_mean',i,sess.run(layer.kernel_posterior.mean(), feed_dict))
                            print('weights_b_layer_stddev',i,sess.run(layer.kernel_posterior.stddev(), feed_dict))
                            print('weights_b_layer_unscale',i, sess.run(tfd.softplus_inverse(layer.kernel_posterior.stddev()), feed_dict) )
                        except AttributeError:
                            continue
                #print('weights_b=',sess.run(model.weights_b.trainable_weights, feed_dict))

            #print(model.weights.get_weights()) # # DEBUG:



            prelosses, postlosses = [], []

        #if (itr!=0) and itr % SAVE_INTERVAL == 0:
        if itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            #_ = sess.run([model.pretrain_op], feed_dict)  #######?????
            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    # task-average algorithm
    if FLAGS.task_average:

        for i in range(len(task_weights_sum)):
            task_weights_sum[i] /= float(average_itr)
            sess.run(tf.assign(model.weights.trainable_weights[i],task_weights_sum[i]))
        print('task_weights_sum=',task_weights_sum)
        print('model.weights=',sess.run(model.weights.trainable_weights,feed_dict))
        print('average_itr=',average_itr)

    if FLAGS.task_b:
        w = sess.run(model.fast_weights_b,feed_dict)
        for i in range(len(w)):
            sess.run(tf.assign(model.weights.trainable_weights[i],w[i]))
        print('fast_weights_b=',w)
        print('model.weights=',sess.run(model.weights.trainable_weights,feed_dict))

    #####

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
#NUM_TEST_POINTS = 100

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None,NUM_TEST_POINTS = 1000):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    #np.random.seed(1)
    #random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            #feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]
            #model.inputa = inputa, model.inputb= inputb,  model.labela= labela, model.labelb= labelb, model.meta_lr= 0.0
            #print(inputa.shape,inputa[0])

            # !!!!!!  this is wrong. By this way we can not input the new
            '''
            model.inputa = inputa
            model.inputb= inputb
            model.labela= labela
            model.labelb= labelb
            '''
            #model.meta_lr= 0.0

            #feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb }
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
            #feed_dict = {model.inputa: np.float32(inputa), model.inputb: np.float32(inputb),  model.labela: np.float32(labela), model.labelb: np.float32(labelb), model.meta_lr: 0.0}
            #feed_dict = {model.inputa: tf.cast(inputa, tf.float32), model.inputb: tf.cast(inputb, tf.float32),  model.labela: tf.cast(labela, tf.float32), model.labelb: tf.cast(labelb, tf.float32), model.meta_lr: 0.0}
        if model.classification:
            #_ = sess.run([model.pretrain_op], feed_dict)  #######?????
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        else:  # this is for sinusoid
            #_ = sess.run([model.pretrain_op], feed_dict)  #######?????
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
            #print('grads_1=',grads_1)
            #print('true_weights=',tw)
            #print('watw=',watw)
            #print('wa=',wa)
            #print('wo=',wo)
            '''
            grads_1 = sess.run(model.grads_1, feed_dict)
            #tw = sess.run(model.true_weights_a, feed_dict)
            #watw = sess.run(model.watw, feed_dict)
            outb = sess.run(model.outb, feed_dict)
            lb = sess.run(model.lb, feed_dict)
            wa = sess.run(model.wa.trainable_weights, feed_dict)
            wo = sess.run(model.wo.trainable_weights, feed_dict)
            print('inputa=',inputa)
            print('labela=',labela)
            print('taskoutputa=',sess.run(model.task_outputa, feed_dict))
            print('taskoutputa_test=',sess.run(model.task_outputa_test, feed_dict))
            print('inputb=',inputb)
            print('labelb=',labelb)
            print('outb=',outb)
            print('outb_last=',sess.run(model.outb_last, feed_dict))
            print('lb=',lb)

            print(sess.run(model.weights.layers[0].kernel_posterior.mean()))
            #print(sess.run(model.weights_cp.layers[0].kernel_posterior.mean()))
            print(sess.run(model.weights_test[0].layers[0].kernel_posterior.mean()))
            #print(sess.run(model.weights_test[0].trainable_weights))
            #print(sess.run(model.weights_test[0].layers[1].kernel_posterior))
            print(model.inputa)
            #print('model.inputa_check=',sess.run(model.inputa_check))

            print('output_weights_test')
            print(sess.run(model.task_outputa))
            print(sess.run(model.task_outputa))
            print(sess.run(model.task_lossa))
            print('output_weights')
            print(sess.run(model.task_outputa_test))
            print(sess.run(model.task_outputa_test))
            '''
            #result = sess.run(model.total_loss1,feed_dict)
            #print(sess.run(model.outputas))
            #print(sess.run(model.weights_output.layers[0].kernel_posterior.mean()))
        #print('TEST:',_)
        #print('result=',result[0],result[-1])

        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    #print(FLAGS.logdir)

    out_filename = os.path.join(FLAGS.logdir ,  exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv')
    out_pkl = os.path.join(FLAGS.logdir ,  exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl')
    #print('out_filename=',out_filename)
    #print('out_pkl=',out_pkl)
    result_pkl = os.path.join(FLAGS.logdir ,  exp_string + '/' + 'results'+ '.pkl')
    pickle.dump([means],
             open(result_pkl, 'wb'))

    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    print(nvidia_smi())
    #config.gpu_options.allow_growth = True
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
            #test_num_updates = 30
        else:
            test_num_updates = 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10
            #test_num_updates = 1  #!!!!

    #FLAGS.meta_batch_size = 1     #### debug !!!!!!!!!!!
    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(2*FLAGS.update_batch_size, FLAGS.meta_batch_size)     #2 for train and test data  a,b

    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                print('initializing data_generator')
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                print('done initializing data_generator')

    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            #random.seed(7)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        #random.seed(8)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        #tf_data_load = True
        num_classes = data_generator.num_classes
        np.random.seed(1)
        random.seed(1)
        if FLAGS.train or FLAGS.datasource == 'sinusoid':  #inputs and labels here will not be used for sinusoid case, only inputa_init is used to initialize the models
            #random.seed(5)
            if 'generate' in dir(data_generator):
                batch_x, batch_y, amp, phase = data_generator.generate()

                if FLAGS.baseline == 'oracle':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    for i in range(FLAGS.meta_batch_size):
                        batch_x[i, :, 1] = amp[i]
                        batch_x[i, :, 2] = phase[i]

                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                #input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
                input_tensors = {'inputa_init': inputa}
         #       print('input_tensors=',inputa)

        #random.seed(6)

        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
          #  print('metaval_input_tensors=',inputa)

  #  print(inputa.shape)

        #input_tensors = None
    #sess = tf.InteractiveSession()
    #tf.global_variables_initializer().run()
    print('stop_grad=',FLAGS.stop_grad)
    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load or FLAGS.datasource == 'sinusoid':
        print('input_tensors=',input_tensors)
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        print('metaval_input_tensors=',metaval_input_tensors)
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()


    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=5)

    #saver = loader = tf.train.Saver({'weights':model.weights.trainable_variables}, max_to_keep=2)


    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr


    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size)  + '.numstep' + str(FLAGS.num_updates)  + '.updatelr' + str(FLAGS.train_update_lr)+'.meta_loss'+str(FLAGS.meta_loss)+'.debug'+str(FLAGS.debug)+'.alias'+FLAGS.alias

    if FLAGS.one_sample:
        exp_string += 'one_sample'
    if FLAGS.pretrain_iterations > 0:
        exp_string += 'pretrain'
    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.test_all:
        if FLAGS.model_file:
            model_file_path = os.path.join(FLAGS.logdir,FLAGS.model_file)
        else:
            model_file_path = os.path.join(FLAGS.logdir,exp_string)

        #model_file = tf.train.latest_checkpoint(model_file_path)
        print(model_file_path)


        for test_iter in range(1000,FLAGS.metatrain_iterations,1000):
            model_file = tf.train.latest_checkpoint(model_file_path)
            model_file = model_file[:model_file.index('model')] + 'model' + str(test_iter)
            #print(model_file_path)
            print(test_iter)
            if model_file:
                ind1 = model_file.index('model')
                resume_itr = int(model_file[ind1+5:])
                #print("Restoring model weights from " + model_file)

                try:
                    saver.restore(sess, model_file)
                    test(model, saver, sess, exp_string, data_generator, test_num_updates,NUM_TEST_POINTS = 100)

                except AttributeError:
                    print('error')

            else:
                print(model_file_path + '   not found')



    else:
        if FLAGS.resume or not FLAGS.train:
            #print(FLAGS.logdir + '/' + exp_string)

            if FLAGS.model_file:
                model_file_path = os.path.join(FLAGS.logdir,FLAGS.model_file)
            else:
                model_file_path = os.path.join(FLAGS.logdir,exp_string)

            model_file = tf.train.latest_checkpoint(model_file_path)
            print(model_file_path)

            '''
            if FLAGS.test_iter > 0:   # usually not used
                model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
            '''
            if model_file:
                ind1 = model_file.index('model')
                resume_itr = int(model_file[ind1+5:])
                print("Restoring model weights from " + model_file)
                saver.restore(sess, model_file)
            else:
                print(model_file_path + '   not found')

        if not FLAGS.construct_only:
            if FLAGS.train:
                #print tf.get_default_graph().as_graph_def()
                #for i, var in enumerate(saver._var_list):
                #    print('Var {}: {}'.format(i, var))
                print('start training')
                train(model, saver, sess, exp_string, data_generator, resume_itr)
            #if FLAGS.test:
            else:
                #print(sess.run(model.weights.layers[0].kernel_posterior.mean()))
                test(model, saver, sess, exp_string, data_generator, test_num_updates)
                #for i, var in enumerate(saver._var_list):
                #    print('Var {}: {}'.format(i, var))

if __name__ == "__main__":
    main()
