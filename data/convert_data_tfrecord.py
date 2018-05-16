import tensorflow as tf
import os  
import numpy as np  
import struct  
import PIL.Image 
from data import cfgs


tf.app.flags.DEFINE_string('save_dir','./tfrecords/','saving dir')
tf.app.flags.DEFINE_string('name',None,'train or test')
FLAGS = tf.app.flags.FLAGS

chars = cfgs.classes

data_dir = {
    'train':'HWDB1.1trn_gnt',
    'test':'HWDB1.1tst_gnt'
}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_from_gnt_dir(gnt_dir):  
    def one_file(f):  
        header_size = 10  
        while True:  
            header = np.fromfile(f, dtype='uint8', count=header_size)  
            if not header.size: break  
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)  
            tagcode = header[5] + (header[4]<<8)  
            width = header[6] + (header[7]<<8)  
            height = header[8] + (header[9]<<8)  
            if header_size + width*height != sample_size:  
                break  
            image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))  
            yield image, tagcode  
   
    for file_name in os.listdir(gnt_dir):  
        if file_name.endswith('.gnt'):  
            file_path = os.path.join(gnt_dir, file_name)  
            with open(file_path, 'rb') as f:  
                for image, tagcode in one_file(f):  
                    yield image, tagcode

def pad(img):
    pad_size = abs(img.shape[0]-img.shape[1]) // 2  
    if img.shape[0] < img.shape[1]:  
        pad_dims = ((pad_size, pad_size), (0, 0))  
    else:  
        pad_dims = ((0, 0), (pad_size, pad_size))  
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)  
    return img

def convert_to_tfrecord():
    gnt_dir = data_dir[FLAGS.name]
    print(gnt_dir)
    save_dir = FLAGS.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = save_dir + FLAGS.name + '.tfrecord'

    train_num = np.zeros([len(chars),])
    writer = tf.python_io.TFRecordWriter(path=save_name)
    for image, tagcode in read_from_gnt_dir(gnt_dir):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        if tagcode_unicode in chars:
            width = image.shape[0]
            height = image.shape[1]
            image_raw = image.tostring()
            label = chars.index(tagcode_unicode)
            features = tf.train.Features(feature={
                'img':_bytes_feature(image_raw),
                'label':_int64_feature(label),
                'img_width':_int64_feature(width),
                'img_height':_int64_feature(height),
            })
            train_num[int(label)] += 1

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
    writer.close()

    print(train_num)
    print(np.sum(train_num))
    print('\nConversion is complete!')


if __name__ == '__main__':
    convert_to_tfrecord()

