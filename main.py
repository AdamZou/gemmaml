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
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_float('sigma', 1.0, 'scale of label distribution')
flags.DEFINE_integer('num_repeat', 1, 'number of repeated runnings for each prediction')
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 2, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 10, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')


## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
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
            #print('inputa.shape=',inputa.shape,inputa)
            #print('inputb.shape=',inputb.shape,inputb)
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations:    
	    input_tensors = [model.pretrain_op]  
        else:
##### metatrain_op
            input_tensors = [model.metatrain_op]
        
        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
        

        result = sess.run(input_tensors, feed_dict)
        #print('result=',result,'\n')
        #print('result=',result[0],'\n')
        #print('result=',result[1],'\n')
        #print('result=',result[2],'\n')
        #print('result=',result[3],'\n') 
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            outb = sess.run(model.outb, feed_dict)
            lb = sess.run(model.lb, feed_dict)
            wa = sess.run(model.wa.trainable_weights, feed_dict)
            wo = sess.run(model.wo.trainable_weights, feed_dict)
            print('wa=',wa)
            print('wo=',wo) 
            print('inputa=',inputa)
            print('labela=',labela)
            print('taskoutputa=',sess.run(model.task_outputa, feed_dict))
            print('taskoutputa_test=',sess.run(model.task_outputa_test, feed_dict))
            print('inputb=',inputb)
            print('labelb=',labelb)
            print('outb=',outb)
            print('outb_last=',sess.run(model.outb_last, feed_dict))
            print('lb=',lb)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
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

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

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
            grads_1 = sess.run(model.grads_1, feed_dict)
            #tw = sess.run(model.true_weights_a, feed_dict)
            #watw = sess.run(model.watw, feed_dict)
            outb = sess.run(model.outb, feed_dict) 
            lb = sess.run(model.lb, feed_dict)
            wa = sess.run(model.wa.trainable_weights, feed_dict)
            wo = sess.run(model.wo.trainable_weights, feed_dict)
            print('TEST:',_)
            print('result=',result)
            #print('grads_1=',grads_1)
            #print('true_weights=',tw)
            #print('watw=',watw)
            #print('wa=',wa)
            #print('wo=',wo)
            print('inputa=',inputa)
            print('labela=',labela)
            print('taskoutputa=',sess.run(model.task_outputa, feed_dict))
            print('taskoutputa_test=',sess.run(model.task_outputa_test, feed_dict))
            print('inputb=',inputb)
            print('labelb=',labelb)
            print('outb=',outb)
            print('outb_last=',sess.run(model.outb_last, feed_dict))
            print('lb=',lb) 
            '''
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
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            #test_num_updates = 5
            test_num_updates = 2
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
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


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
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
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
        if FLAGS.train or FLAGS.datasource == 'sinusoid':
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
    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load or FLAGS.datasource == 'sinusoid':
        print('input_tensors=',input_tensors)
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        print('metaval_input_tensors=',metaval_input_tensors)
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=5)
    #saver = loader = tf.train.Saver(model.weights.trainable_variables, max_to_keep=2)


    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)
    
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
  
    ''' 
    print('test_1:') 
    print(tf.cast(inputa[0], tf.float32))
    print(sess.run(model.weights.layers[1].kernel_posterior.mean()))
    #print(sess.run(model.weights_cp.layers[0].kernel_posterior.mean()))
    print(sess.run(model.weights_test[0].layers[1].kernel_posterior.mean()))
    print(model.inputa) 
    print('model.inputa_check=',model.inputa_check)
    print('output_weights_test')
    print(sess.run(model.task_outputa))
    print(sess.run(model.task_outputa))
    print('output_weights')
    print(sess.run(model.task_outputa_test))
    print(sess.run(model.task_outputa_test))
    '''

    for i, layer in enumerate(model.weights_test[0].layers):
        print(i)
        try:
            print('layer',i)
            print(sess.run(layer.kernel_posterior.mean()))
            print(sess.run(layer.kernel_posterior.variance()))
            print(sess.run(layer.bias_posterior.mean()))
            print(sess.run(layer.bias_posterior.variance()))
        except AttributeError:
            continue

    inputa_1 = np.array([[[-4.99885625],
        [-1.97667427],
        [-3.53244109],
        [-4.07661405],
        [-3.13739789],
        [-1.54439273],
        [-1.03232526],
        [ 0.38816734],
        [-0.80805486],
        [ 1.852195  ]]])
    print(sess.run(model.weights_test[0](tf.cast(inputa_1[0], tf.float32))))
    print(sess.run(model.weights_test[0](tf.cast(inputa_1[0], tf.float32))))

    if FLAGS.resume or not FLAGS.train:
        print(FLAGS.logdir + '/' + exp_string)
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
    
    #print(inputa[0])
    for i, layer in enumerate(model.weights_test[0].layers):
        print(i)
    	try: 
            print('layer',i)
            print(sess.run(layer.kernel_posterior.mean()))
            print(sess.run(layer.kernel_posterior.variance()))
            print(sess.run(layer.bias_posterior.mean()))
            print(sess.run(layer.bias_posterior.variance()))
        except AttributeError:
            continue

    inputa_1 = np.array([[[-4.99885625],
        [-1.97667427],
        [-3.53244109],
        [-4.07661405],
        [-3.13739789],
        [-1.54439273],
        [-1.03232526],
        [ 0.38816734],
        [-0.80805486],
        [ 1.852195  ]]])
    print(sess.run(model.weights_test[0](tf.cast(inputa_1[0], tf.float32))))
    print(sess.run(model.weights_test[0](tf.cast(inputa_1[0], tf.float32))))
    #print(sess.run(model.weights_test(inputa[0])))
    '''
    print('test_2:')
    print(sess.run(model.weights.layers[0].kernel_posterior.mean()))
    #print(sess.run(model.weights_cp.layers[0].kernel_posterior.mean()))
    print(sess.run(model.weights_test[0].layers[0].kernel_posterior.mean()))
    #print(sess.run(model.weights_test(inputa[0])))
    print(model.inputa)
    print('model.inputa_check=',model.inputa_check)
    print('output_weights_test')
    print(sess.run(model.task_outputa))
    print(sess.run(model.task_outputa))
    print(sess.run(model.task_lossa))
    print('output_weights')
    print(sess.run(model.task_outputa_test))
    print(sess.run(model.task_outputa_test)) 
    '''
    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
       
        #print(sess.run(model.weights.layers[0].kernel_posterior.mean()))
        test(model, saver, sess, exp_string, data_generator, test_num_updates) 
        '''
        print('test_3:')
        print(sess.run(model.weights.layers[0].kernel_posterior.mean()))
        #print(sess.run(model.weights_cp.layers[0].kernel_posterior.mean()))
        print(sess.run(model.weights_test[0].layers[0].kernel_posterior.mean())) 
        #print(sess.run(model.weights_test[0].trainable_weights))
        #print(sess.run(model.weights_test[0].layers[1].kernel_posterior))   
        print('model.inputa=',model.inputa)  
        print('model.inputa_check=',model.inputa_check) 
        print('output_weights_test')
        print(sess.run(model.task_outputa))
        print(sess.run(model.task_outputa))
        print(sess.run(model.task_lossa))
        print('output_weights')
        print(sess.run(model.task_outputa_test))
        print(sess.run(model.task_outputa_test))
        '''
if __name__ == "__main__":
    main()

