
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from utils.timer import Timer
from data import cfgs
from data.convert_data_tfrecord import pad
from networks.get_network import get_network_byname
from networks.losses import cross_losses
from data.read_tfrecord import next_batch
#from data.read_tfrecord_to_list import test_data

slim = tf.contrib.slim

os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.GPUs

tf.app.flags.DEFINE_string('weights',None,'choose weights to restore')
tf.app.flags.DEFINE_string('net_name','ournet','choose net to evaluate')
tf.app.flags.DEFINE_string('image',None,'choose test image')
FLAGS = tf.app.flags.FLAGS



def test():
    test_time = Timer()
    image = Image.open(FLAGS.image).convert('L')
    image = image.resize((cfgs.image_size,cfgs.image_size),Image.ANTIALIAS)
    image = np.asarray(image,dtype=float)
    image = (image - np.ones_like(image)) / 255
    image = np.reshape(image,[1,cfgs.image_size,cfgs.image_size,1])
    

    
    with tf.Graph().as_default():
        
        img = tf.placeholder(tf.float32,[1,cfgs.image_size,cfgs.image_size,1])
        

        _,end_points = get_network_byname(name=cfgs.net_name,
                                          inputs=img,
                                          num_classes=len(cfgs.classes),
                                          is_training=False)

        prediction = end_points['prediction']
        pre = tf.arg_max(prediction,1)

        conv1 = tf.expand_dims(end_points['conv1'][:,:,:,10],axis=3)
        conv2 = tf.expand_dims(end_points['conv2'][:,:,:,10],axis=3)
        block1 = tf.expand_dims(end_points['block1'][:,:,:,10],axis=3)
#        tf.summary.image('conv1',conv1)
#        tf.summary.image('conv2',conv2)
#        tf.summary.image('block1',block1)

#        summary_op = tf.summary.merge_all()
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


#            summary_path = os.path.join('test_summary',cfgs.net_name)

#            if not os.path.join(summary_path):
#                os.mkdir(summary_path)
#            summary_writer = tf.summary.FileWriter(summary_path,tf.get_default_graph())

            
            test_time.tic()
            _pre = sess.run([pre],feed_dict={img:image})
            test_time.toc()

#            summary_str = sess.run(summary_op)
#            summary_writer.add_summary(summary_str)
#            summary_writer.flush()

#            summary_writer.close()

            print('this image is : {}\ncost time : {}'.format(cfgs.classes[int(_pre[0])],test_time.average_time))



if __name__=='__main__':
    test()



