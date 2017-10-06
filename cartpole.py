#!/usr/bin/python
#coding: utf-8
import numpy as np

import tensorflow as tf
from rl_tools import Memory, DQN, greedy, boltzman, select_action_randomly, create_copy_weights, update_weights, create_update_target_graphs



#======= ALL CONSTANTS ARE LOCATED HERE =====
# Learning_rate, eps, update_rate etc...
#========================================
from const import *

import gym
env = gym.make('CartPole-v1') #Works only on v1 at the moment, don't change this
observation = env.reset()
env.render()

target_net = DQN("target")
main_net = DQN("main")

trainables = tf.trainable_variables()
copy_weights = create_copy_weights(trainables)
target_updater = create_update_target_graphs(trainables)

total_actions_done = 0
last_total = []

if EXPLOR == 'eps':
    eps = EPS_BEGIN
    select_action = greedy
else:
    select_action = boltzman
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    update_weights(copy_weights,sess)

    test1 = np.array([0.01,0.01,0.01,0.01]).reshape((1,ENV_SIZE))
    res1 = sess.run([main_net.forward],feed_dict={main_net.state_in:test1})[0][0]
    res2 = sess.run([target_net.forward],feed_dict={target_net.state_in:test1})[0][0]
    assert np.array_equal(res1,res2)
    # Check if both network shares same weights
    
    memory = Memory(MEMORY_SIZE)

    for ep in range(NUM_EP):
        observation = env.reset()
        total_reward = 0
        done = False
        num_action_for_this_ep = 0
        while not done :
            observation = observation.reshape((1,ENV_SIZE))

            if EXPLOR=='eps'and np.random.random(1)[0]<eps:
                action = select_action_randomly()
                eps = max(eps-EPS_STEP, EPS_END)
            else: # Softmax aka boltzman
                # batch of size 1, so you take the first elem [0]
                out_value = sess.run([main_net.forward],feed_dict={main_net.state_in:observation})[0][0]
                action = select_action(out_value)

                # if total_actions_done % 100 == 0:
                #     print("out_value :",out_value)

            
            next_observation, reward, done, info = env.step(action)
            total_actions_done +=1
            num_action_for_this_ep += 1

            if done:
                if num_action_for_this_ep>=499:
                    print("WIN, 500")
                    memory.add_entry([observation[0], action, next_observation,1])
                else:
                    memory.add_entry([observation[0], action, next_observation,0])
                last_total.append(total_reward)

                if ep%PRINT_MEAN==PRINT_MEAN-1:
                    mean = sum(last_total)/len(last_total)
                    print("EP N°{} : MEAN FOR LAST 10 episodes : {}".format(ep,mean))
                    if EXPLOR=='eps':
                        print("eps = ",eps)
                    last_total = []

            else:
                memory.add_entry([observation[0], action, next_observation, reward])
                total_reward += reward

            #reset observation
            observation = next_observation

            #optimize network
            if total_actions_done>=BATCH_SIZE*2:

                batch = memory.sample(BATCH_SIZE)

                current_states = np.vstack(batch[:,0])
                next_states = np.vstack(batch[:,2])
                rewards = batch[:,3]
                actions = np.array(batch[:,1],dtype=int)
                #Reformating input so it can be taken as input for tensorflow gather_nd function
                actions = np.concatenate((np.arange(BATCH_SIZE), actions), axis=0).reshape(2,BATCH_SIZE).T


                next_state_q = target_net.forward.eval(feed_dict={target_net.state_in:next_states})
                next_state_q = next_state_q.max(1)

                ref_next_q = next_state_q*DISCOUNT_FACT + reward
                ref_next_q[np.where(rewards==0)] = 0

                #print("ref_next_q",ref_next_q)
                #input()
                
                #===== PRINTING GRADIENT NORM =================
                # print(main_net.grad_global_norm.eval(feed_dict={
                #     main_net.state_in:current_states,
                #     main_net.true_q:ref_next_q,
                #     main_net.actions_taken:actions
                # }))
                # input()
                
                main_net.train_step.run(feed_dict={
                    main_net.state_in:current_states,
                    main_net.true_q:ref_next_q,
                    main_net.actions_taken:actions
                })


                if ep%TARGET_UPDATE==0:
                    update_weights(target_updater, sess)
                
            env.render()


        
