import tensorflow as tf
    
slim = tf.contrib.slim
    
def simple_net_arg_scope(weight_decay=0.00005,
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
            'fused':None,
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
            return scope

    
    
    
def simple_net(inputs,num_classes=None,is_training=True):

    end_points = {} 

    keep_prob = 0.5

    with tf.variable_scope('simple'):
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                                stride=1,padding='SAME'):
                net = slim.conv2d(inputs,32,3,scope='conv1')
                net = slim.max_pool2d(net,2,stride=2,scope='max_pool1')
                net = slim.conv2d(net,64,3,scope='conv2')
                net = slim.max_pool2d(net,2,stride=2,scope='max_pool2')
                net = slim.conv2d(net,80,3,scope='conv3')
                net = slim.max_pool2d(net,2,stride=2,scope='max_pool3')
                net = slim.conv2d(net,96,3,scope='conv4')
                kernel_size = net.get_shape()[1:3]
                net = slim.avg_pool2d(net,kernel_size,padding='VALID',scope='avg_pool')
                net = slim.flatten(net)
                net = slim.fully_connected(net,1024,scope='fully_connected')
                net = slim.dropout(net,keep_prob,is_training=is_training)
                net = slim.fully_connected(net,num_classes,scope='logits')
                end_points['logits'] = net
                end_points['prediction'] = tf.nn.softmax(net)

        return net,end_points 

