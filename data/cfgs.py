import os
HOME = os.path.expanduser("~")
path = os.path.join(HOME,'PycharmProjects/recognization')

classes = '武汉大学数统毕业论文'
image_size = 64

batch_size = 32
net_name='ournet'
learning_rate=0.001
max_iteration=150

GPUs = '1'

#test num
num = 596