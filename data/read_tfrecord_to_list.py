import tensorflow as tf
from data import cfgs
import numpy as np

class test_data():
    def __init__(self,num,test_file):
        self.total = np.zeros([len(cfgs.classes),])
        self.get_label = []
        img,index = self.prepare(test_file)
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(num):
            _img,_index = sess.run([img,index])
            gt = {'img':_img,'label':_index}
            self.get_label.append(gt)
            self.total[_index] += 1
            


    def prepare(self,test_file):
        reader = tf.TFRecordReader()
        file_queue = tf.train.string_input_producer([test_file])
        _,serialized_example = reader.read(file_queue)
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
        index = tf.cast(features['label'], tf.int64)
        img = tf.cast(img,tf.float32)
        img = (img - 128 * tf.ones_like(img)) / 255
        img = tf.expand_dims(img,axis=0)
        img = tf.image.resize_bilinear(img,[cfgs.image_size,cfgs.image_size])
        img = tf.reshape(img,[1,cfgs.image_size,cfgs.image_size,1])
        return img,index