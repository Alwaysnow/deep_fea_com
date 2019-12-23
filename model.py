import tensorflow as tf 
import numpy as np 

def encoder_non(input):
    with tf.variable_scope("fea_non"):
        output_1 = tf.layers.conv2d(input, 64, [1,1], strides=[1,1], padding='same')
        output_1 = tf.contrib.layers.gdn(output_1)
        output_2 = tf.layers.conv2d(output_1, 64, [1,1], strides=[1,1], padding='same')
        return output_2

def decoder_non(input):
    with tf.variable_scope("fea_non"):
        output_1 = tf.layers.conv2d(input, 128, [1,1], strides=[1,1], padding='same')
        output_1 = tf.contrib.layers.gdn(output_1, inverse=True)
        output_2 = tf.layers.conv2d(output_1, 128, [1,1], strides=[1,1], padding='same')
        return output_2
