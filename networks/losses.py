import tensorflow as tf

slim = tf.contrib.slim

def cross_losses(labels,logits):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
