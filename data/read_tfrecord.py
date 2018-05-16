from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import os
from data import cfgs
from PIL import Image
import matplotlib.pyplot as plt

tfrecord_path = os.path.join(cfgs.path,'data/tfrecords')

def read_and_resize(filename_queue,image_size):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64),
            'img_width':tf.FixedLenFeature([],tf.int64),
            'img_height':tf.FixedLenFeature([],tf.int64)
        }
    )
    img = tf.decode_raw(features['img'], tf.uint8)
    img_width = tf.cast(features['img_width'], tf.int32)
    img_height = tf.cast(features['img_height'], tf.int32)
    img = tf.reshape(img,[img_width,img_height,1])
    index = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(index,len(cfgs.classes))

    img = tf.cast(img,tf.float32)
    img = (img - 128 * tf.ones_like(img)) / 255
    img = tf.expand_dims(img,axis=0)
    img = tf.image.resize_bilinear(img,[image_size,image_size])
    img = tf.squeeze(img,axis=0)
    img = tf.image.random_brightness(img,max_delta=0.3)
    return img,label,img_width,img_height



def next_batch(batch_size,image_size,is_training,is_shuffle=True):
    if is_training:
        pattern = os.path.join(tfrecord_path,'train.tfrecord')
        print(pattern)
    else:
        pattern = os.path.join(tfrecord_path,'test.tfrecord')
    
    filename_tensorlist = tf.train.match_filenames_once(pattern)

    filename_queue = tf.train.string_input_producer(filename_tensorlist)

    img,label,img_width,img_height = read_and_resize(filename_queue,image_size)

    if is_shuffle:
        img_batch,label_batch = tf.train.shuffle_batch([img,label],
                                                       batch_size=batch_size,
                                                       capacity=200,
                                                       min_after_dequeue=100,
                                                       num_threads=4)
    else:
        img_batch,label_batch = tf.train.batch([img,label],
                                               batch_size=batch_size,
                                               num_threads=1,
                                               capacity=100)
    return img_batch,label_batch


def test():
    img,label = next_batch(cfgs.batch_size,cfgs.image_size,is_training=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        theads = tf.train.start_queue_runners(sess,coord)
        for i in range(10):
            im,la = sess.run([img,label])
            print(im.shape)
            if i == 2:
                im = np.array(im,np.int32)
                im = np.squeeze(im)
                print(im.shape)
                plt.imshow(im,cmap='gray')
                plt.show()
                print(np.argmax(la))
        coord.request_stop()
        coord.join(theads)

if __name__=='__main__':
    test()