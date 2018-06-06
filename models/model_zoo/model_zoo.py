# Copyright 2018 franksai. All Rights Reserved.
# ==============================================================================
# you can find origin model in tensorflow repo
# https://github.com/tensorflow/models/tree/master/research/slim/nets
# files in "./networks" is easier to import ; sample as inception


import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception as inception 

from resnet_v1 import resnet_v1_50


def resnet(images,OUTPUT_CLASS):
    logits,_ = resnet_v1_50(images ,num_classes=OUTPUT_CLASS )
    return logits

def wnet(images,OUTPUT_CLASS):
    net = slim.conv2d(images, 20, [10,10], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 20, [7,7], scope='conv2')
    net = slim.conv2d(net, 20, [5,5], scope='conv3')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.conv2d(net, 50, [5,5], scope='conv4')
    net = slim.max_pool2d(net, [2,2], scope='pool3')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.dropout(net, 0.5, is_training=True,scope='dropout5')
    net = slim.fully_connected(net, OUTPUT_CLASS, activation_fn=None, scope='fc6')
    return net

def lenet(images,OUTPUT_CLASS):
    net = slim.conv2d(images, 20, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 50, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.fully_connected(net, OUTPUT_CLASS, activation_fn=None, scope='fc5')
    return net


def vgg16(images,OUTPUT_CLASS):
    fc_conv_padding='VALID'
    is_training=True
    dropout_keep_prob=0.5
    net = slim.repeat(images, 1, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    # Use conv2d instead of fully_connected layers.
    net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout6')
    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout7')
    net = slim.conv2d(net, OUTPUT_CLASS, [1, 1], activation_fn=None,normalizer_fn=None,scope='fc8')
    return net
