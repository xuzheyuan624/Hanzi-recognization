import tensorflow as tf

slim = tf.contrib.slim
from networks import ournet
from networks import simple_net



def get_network_byname(name,
                       inputs,
                       num_classes=None,
                       is_training=True):
    if name == 'ournet':
        with slim.arg_scope(ournet.ournet_arg_scope()):
            net,end_points = ournet.ournet(inputs,num_classes,is_training=is_training)

        return net,end_points
    elif name == 'simple':
        with slim.arg_scope(simple_net.simple_net_arg_scope()):
            net,end_points = simple_net.simple_net(inputs,num_classes=num_classes,is_training=is_training)
        return net,end_points