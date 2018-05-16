from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
from utils.timer import Timer
from data import cfgs
from networks.get_network import get_network_byname
from networks.losses import cross_losses
from data.read_tfrecord import next_batch

slim = tf.contrib.slim

os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.GPUs

tf.app.flags.DEFINE_string('restore_dir',None,'choose weights to restore')
tf.app.flags.DEFINE_string('save_dir','weights','dir to save weights')
FLAGS = tf.app.flags.FLAGS

def train():
    train_time = Timer()

    with tf.Graph().as_default():
        with tf.name_scope('get_batch'):
            img_batch,gt_batch = next_batch(cfgs.batch_size,cfgs.image_size,is_training=True,is_shuffle=True)


        _,end_points = get_network_byname(name=cfgs.net_name,
                                            inputs=img_batch,
                                            num_classes=len(cfgs.classes),
                                            is_training=True)

        prediction = end_points['prediction']
        logits = end_points['logits']
#        cross_losses(logits,gt_batch)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_batch,logits=logits))
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()
        correct_prediction = tf.equal(tf.arg_max(prediction,1),tf.arg_max(gt_batch,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        global_step = slim.get_or_create_global_step()

        lr = tf.train.piecewise_constant(global_step,
                                         boundaries=[np.int64(5000), np.int64(10000)],
                                         values=[cfgs.learning_rate, cfgs.learning_rate / 10, cfgs.learning_rate / 100])
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = slim.learning.create_train_op(total_loss,optimizer,global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_loss = control_flow_ops.with_dependencies([updates],total_loss)
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('accuracy',accuracy)
        summary_op = tf.summary.merge_all()
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer = tf.train.Saver(tf.global_variables(),max_to_keep=6)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=6)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth =True
        # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
        #内存，所以会导致碎片
        with tf.Session(config=config) as sess:
            sess.run(init_op)

            if not FLAGS.restore_dir is None:
                restorer.restore(sess,FLAGS.restore_dir)

            coord = tf.train.Coordinator()
            theads = tf.train.start_queue_runners(sess,coord)

            summary_path = os.path.join(cfgs.path,'summary/' + cfgs.net_name)
            if not os.path.exists(summary_path):
                os.mkdir(summary_path)
            summary_writer = tf.summary.FileWriter(summary_path,tf.get_default_graph())

            for step in range(cfgs.max_iteration):
                train_time.tic()

                _lr,_global_step,_loss,_accuracy,_ = sess.run([lr,global_step,total_loss,accuracy,train_op])

                train_time.toc()

                if step %  10 == 0:
                    print('step:{},loss:{},accuracy:{},\nLearning rate:{},Speed:{},Remain:{}'
                          .format(_global_step,_loss,_accuracy,_lr,train_time.average_time,train_time.remain(step,cfgs.max_iteration)))

                if step % 10 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str,step)
                    summary_writer.flush()

                if step % 1000 == 0:
                    save_dir = os.path.join(cfgs.path,FLAGS.save_dir)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_ckpt = os.path.join(save_dir,cfgs.net_name + '_' + str(_global_step)+'_model.ckpt')
                    saver.save(sess,save_ckpt)
                    print('weights had been saved')
            final_save = os.path.join(save_dir,cfgs.net_name + "_final.ckpt")
            saver.save(sess,final_save)
            print('Having saved final weights')

            summary_writer.close()
            
            coord.request_stop()
            coord.join(theads)


if __name__=='__main__':
    train()
