#!/usr/bin/python
#coding: utf-8

import numpy as np
import tensorflow as tf
from tf_tools import *
from const import *

class Memory(object):
    def __init__(self, max_size):
        self.content=np.empty((max_size,4),dtype=object)
        #memory content, column in order : state_t0, action, state_t1, reward
        self.position=0 #to keep track of the memory when it's not full
        self.current_size = 0
        self.max_size = max_size
    def add_entry(self, entry):
        self.content[self.position] = entry
        self.position = (self.position+1)%self.max_size
        self.current_size += 1

    def sample(self, batch_size):
        size = min(self.current_size,self.max_size)
        random_sample = np.random.choice(size,batch_size)
        batch = self.content[random_sample]
        return batch

class DQN(object):
    def __init__(self,name='default'):

        with tf.name_scope(name): 
            self.state_in = tf.placeholder(tf.float32, shape=[None,ENV_SIZE])
            self.true_q = tf.placeholder(tf.float32)
            self.actions_taken = tf.placeholder(tf.int32, shape=[BATCH_SIZE,2])

            self.hidden_layer= linear_layer(self.state_in,ENV_SIZE,NUM_HIDDEN,with_act=True,name='hidden_layer')
            self.out = linear_layer(self.hidden_layer,NUM_HIDDEN,NUM_ACTION,with_act=False,name='out')

            self.q_selected = tf.gather_nd(self.out, self.actions_taken)

            #self.mse_loss = tf.losses.huber_loss(self.true_q,self.q_selected)
            self.mse_loss = tf.losses.mean_squared_error(self.true_q,self.q_selected)

            self.optim = tf.train.AdamOptimizer(LR)

            gradients, variables = zip(*self.optim.compute_gradients(self.mse_loss))

            gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
            #gradients, _ = tf.clip_by_value(gradients, -1.0, 1.0)
            self.grad_global_norm = tf.global_norm(gradients)

            self.train_step = self.optim.apply_gradients(zip(gradients,variables))
            self.forward = self.out


def greedy(value_array):
    return np.argmax(softmax(value_array)) #function in tf_tools

def boltzman(value_array):
    proba = softmax(value_array)
    return np.random.choice(NUM_ACTION, p=proba)

def select_action_randomly(*args,**kwargs):
    action = np.random.choice(NUM_ACTION)
    return action

def create_copy_weights(trainables):
    num_var,odd = divmod(len(trainables),2)
    assert not odd, "Problem with variables, odd number of variables, it should be same networks, so 2 times the same variables."

    copy_var_updater = []
    for i,var in enumerate(trainables[0:num_var]):
        copy_var_updater.append(
            var.assign(trainables[i+num_var].value())
        )
    return copy_var_updater
    
def create_update_target_graphs(trainables):
    num_var,odd = divmod(len(trainables),2)
    assert not odd, "Problem with variables, odd number of variables, it should be same networks, so 2 times the same variables."

    target_var_updater = []
    for i,var in enumerate(trainables[0:num_var]):
        target_var_updater.append(
            var.assign(  (1-TAU)*var.value() + TAU*trainables[i+num_var].value() )
            )
    return target_var_updater

def update_weights(updater,sess):
    for weight_updater in updater:
        sess.run([weight_updater])
