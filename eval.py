from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
import numpy as np
from utils.timer import Timer
from data import cfgs
from networks.get_network import get_network_byname
from networks.losses import cross_losses
from data.read_tfrecord import next_batch
#from data.read_tfrecord_to_list import test_data

slim = tf.contrib.slim

os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.GPUs

tf.app.flags.DEFINE_string('weights',None,'choose weights to restore')
tf.app.flags.DEFINE_string('net_name','ournet','choose net to evaluate')
tf.app.flags.DEFINE_string('test_file','data/tfrecords/test.tfrecord','choose test file')
FLAGS = tf.app.flags.FLAGS



def eval():
    test_time = Timer()
    pre_correct = np.zeros([len(cfgs.classes)])
    total = np.zeros([len(cfgs.classes)])
#    gt_label = data.get_label


    with tf.Graph().as_default():
        with tf.name_scope('get_batch'):
            img_batch,gt_batch = next_batch(1,cfgs.image_size,is_training=False,is_shuffle=False)
#        img = tf.placeholder(tf.float32,[1,cfgs.image_size,cfgs.image_size,1])
#        label = tf.placeholder(tf.int64,[])

        _,end_points = get_network_byname(name=cfgs.net_name,
                                            inputs=img_batch,
                                            num_classes=len(cfgs.classes),
                                            is_training=False)

        prediction = end_points['prediction']
        pre = tf.arg_max(prediction,1)
        label = tf.arg_max(gt_batch,1)
        correct = tf.equal(pre,label)

        conv1 = tf.expand_dims(end_points['conv1'][:,:,:,10],axis=3)
        conv2 = tf.expand_dims(end_points['conv2'][:,:,:,10],axis=3)
        block1 = tf.expand_dims(end_points['block1'][:,:,:,10],axis=3)
        tf.summary.image('conv1',conv1)
        tf.summary.image('conv2',conv2)
        tf.summary.image('block1',block1)

        summary_op = tf.summary.merge_all()
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        model_variables = slim.get_model_variables()
        restore_variables = [var for var in model_variables if var.name.startswith(cfgs.net_name)]

        restorer = tf.train.Saver(restore_variables,max_to_keep=6)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth =True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not FLAGS.weights:
                raise ValueError('please input a correct weights')
            else:
                restorer.restore(sess,FLAGS.weights)
                print('restore weights from {}'.format(FLAGS.weights))

            coord = tf.train.Coordinator()
            theads = tf.train.start_queue_runners(sess, coord)

            summary_path = os.path.join('eval',cfgs.net_name)

            if not os.path.join(summary_path):
                os.mkdir(summary_path)
            summary_writer = tf.summary.FileWriter(summary_path,tf.get_default_graph())

            for i in range(int(cfgs.num)):
#                image = gt_label[i]['img']
#                l = gt_label[i]['label']
#                feed_dict = {img: image, label: l}
                test_time.tic()
                _label,_pre,_correct = sess.run([label,pre,correct])
                test_time.toc()
                
                total[_label[0]] += 1
                if _correct[0]:
                    pre_correct[_pre[0]] += 1
                
#                if _correct:
#                    correct[int(_pre)] += 1
                if i % 10 == 0:
                    print('Having evaluated {}/{}\n{} is predicted as {}\ncost time:{}'
                          .format(i,cfgs.num,cfgs.classes[int(_label[0])],cfgs.classes[int(_pre[0])],test_time.average_time))
                if i % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str,i)
                    summary_writer.flush()

            summary_writer.close()
            coord.request_stop()
            coord.join(theads)
        
        accuracy = pre_correct / total
        all_accuracy = np.sum(pre_correct) / np.sum(total)
        print(all_accuracy)
        result_path = 'results'
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        result_file = os.path.join(result_path,cfgs.net_name + '.txt')
        with open(result_file,'w') as f:
            for j in range(len(cfgs.classes)):
                f.write('accuracy of {} is : {} \n'.format(cfgs.classes[j],accuracy[j]))
            f.write('total accuracy is : %.3f'%all_accuracy)



if __name__=='__main__':
    eval()

            