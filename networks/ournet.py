from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def ournet_arg_scope(weight_decay=0.00005,
                     stddev=0.01,
                     batch_norm_decay=0.9997,
                     batch_norm_epsilon=0.001,
                     activation_fn=tf.nn.relu):
    """
    return the scope with the default parameters for ournet
    """
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_regularizer=slim.l2_regularizer(weight_decay)):
        batch_norm_params = {
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'updates_collections':tf.GraphKeys.UPDATE_OPS,
            'fused':None
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
            return scope

def block1(net,scale=1.0,activation_fn=tf.nn.relu,scope=None,reuse=None):
    with tf.variable_scope(scope,'block1',[net]):
        with tf.variable_scope('branch_0'):
            branch_0 = slim.conv2d(net,32,1,scope='conv1')
        with tf.variable_scope('branch_1'):
            branch_1 = slim.conv2d(net,32,1,scope='conv1')
            branch_1 = slim.conv2d(branch_1,48,[1,3],scope='conv2')
            branch_1 = slim.conv2d(branch_1,64,[3,1],scope='conv3')
        mixed = tf.concat([branch_0,branch_1],axis=3)
        up = slim.conv2d(mixed,net.get_shape()[3],1,normalizer_fn=None,
                                                    activation_fn=None,
                                                    scope='conv1')
        scaled_up = scale * up
        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net

def inception_pool(net,activation_fn=tf.nn.relu,scope=None,reuse=None):
    with tf.variable_scope(scope,'inception_pool',[net]):
        with tf.variable_scope('branch_0'):
            branch_0 = slim.conv2d(net,32,1,scope='conv1')
            branch_0 = slim.max_pool2d(branch_0,3,stride=2,scope='max_pool2d')
        with tf.variable_scope('branch_1'):
            branch_1 = slim.conv2d(net,32,1,scope='conv1')
            branch_1 = slim.conv2d(branch_1,48,3,stride=2,scope='conv2')
        net = tf.concat([branch_0,branch_1],axis=3)
    return net


def ournet(inputs,
           num_classes=None,
           activation_fn=tf.nn.relu,
           scope=None,
           is_training=True):
    end_points = {}
    dropout_keep_prob = 0.8

    with tf.variable_scope(scope,'ournet',[inputs]):
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                                stride=1,padding='SAME'):
                net = slim.conv2d(inputs,32,3,scope='conv1')
                end_points['conv1'] = net
                net = slim.max_pool2d(net,2,stride=2,scope='maxpool1')
                end_points['maxpool1'] = net
                net = slim.conv2d(net,64,3,scope='conv2')
                end_points['conv2'] = net
                net = slim.max_pool2d(net,2,stride=2,scope='maxpool2')
                end_points['maxpool2'] = net
                net = block1(net)
                end_points['block1'] = net
                net = inception_pool(net)
                end_points['incep_pool'] = net
                net = slim.conv2d(net,96,3,scope='conv3')
                end_points['conv3'] = net
                if not num_classes:
                    return net, end_points
                kernel_size = net.get_shape()[1:3]
                net = slim.avg_pool2d(net,kernel_size,padding='VALID',scope='global_pool')
                end_points['global_pool'] = net
                net = slim.flatten(net)
                net = slim.fully_connected(net,1024,activation_fn=tf.nn.relu,scope='fully_connected')
                net = slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout')
                net = slim.fully_connected(net,num_classes,activation_fn=None,scope='logits')
                end_points['logits'] = net
                end_points['prediction'] = tf.nn.softmax(net,name='prediction')
        return net, end_points

