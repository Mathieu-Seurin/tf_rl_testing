#!/usr/bin/python
#coding: utf-8
import tensorflow as tf
import numpy as np

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def linear_layer(input_tensor, size_in,size_out, with_act, name=''):

    with tf.name_scope(name):
        w_shape = (size_in, size_out)

        with tf.name_scope('weights'):
            #w = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1))
            w = tf.Variable(tf.random_uniform(w_shape, -1,1))
            variable_summaries(w)

        with tf.name_scope('biases'):
            b = tf.Variable(tf.constant(0.1, shape=[size_out]))
            variable_summaries(b)

        if with_act:
            linear_node = tf.nn.relu(tf.matmul(input_tensor,w)+b)
        else:
            linear_node = tf.matmul(input_tensor,w)+b
        return linear_node

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out
