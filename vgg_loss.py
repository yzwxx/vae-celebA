import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os, sys, time
import numpy as np
from scipy.misc import imread, imresize

def conv_layers_simple_api(net_in, reuse=True):
    # with tf.name_scope('preprocess') as scope:
        
    #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, 
    #             shape=[1, 1, 1, 3], name='img_mean')
    #     net_in.outputs = net_in.outputs - mean
    with tf.variable_scope("vgg_conv", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        l1 = network.outputs
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool1')
        

        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        l2 = network.outputs
        network = Conv2d(network,n_filter=128, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool2')
        

        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        l3 = network.outputs
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool3')
        

        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        l4 = network.outputs
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool4')
        

        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        l5 = network.outputs
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool5')
        

    return network, l1, l2, l3, l4, l5


def fc_layers(net,reuse=True):
    with tf.variable_scope("vgg_fc", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = FlattenLayer(net, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
    return network


def build_vgg(x):
    net_in = InputLayer(x, name='input1')
    net_cnn,_,_,_,_,_ = conv_layers_simple_api(net_in) 
    network = fc_layers(net_cnn)
    y = network.outputs
    probs = tf.nn.softmax(y)
    return network, probs

def main():
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    network, probs = build_vgg(x)
    # network2, probs2 = build_vgg(x)
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    network.print_params()
    network.print_layers()


    npz = np.load('vgg16_weights.npz')
    params = []
    for val in sorted( npz.items() ):
        print("  Loading %s" % str(val[1].shape))
        params.append(val[1])
    tl.files.assign_params(sess, params, network)

    img1 = imread('laska.png', mode='RGB') 
    img1 = imresize(img1, (224, 224))

    prob = sess.run(probs, feed_dict={x: [img1]})[0]
    print(prob)


if __name__ == '__main__':
    main()
