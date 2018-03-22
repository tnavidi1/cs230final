# Pre-training actor network on heuristic

#import math
import numpy as np
#import h5py
#import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.framework import ops

from DDPGModels import *
from powerflowEnv import *

import time

restore_model = True

print('Loading Data')
DataDict = np.load('Data_Processed.npz')

X_train=DataDict['X_train']
Y_train=DataDict['Y_train']

X_dev=DataDict['X_dev']
Y_dev=DataDict['Y_dev'] # load optimal control for dev
Yh_dev = Y_train[:,0:Y_dev.shape[1]] # get hueristic control for the dev set
X_test=DataDict['X_test']
Y_test=DataDict['Y_test'] # load optimal control for test
Yh_test = Y_train[:,0:Y_test.shape[1]] # get hueristic control for the test set

# size of state and action
n_x = X_train.shape[0] # 5 nodes * 24 hours
n_y = Y_train.shape[0]

num_epochs = 4000 #2000
costs=[]

# size of 2 hidden layers in network
layers = [100,50]
layers_critic = [100,50,25,1] #hidden units for state,action,layer2 combine state and action, output

tf.reset_default_graph()

#actor = Actor(n_x, n_y, layers)
actorAsync = AsyncNets(n_x, n_y, layers, class_name='Actor',pretrain=True)
actor,actor_target = actorAsync.get_subnets()
criticAsync = AsyncNets(n_x, n_y, layers_critic, class_name='Critic')
critic,critic_target = criticAsync.get_subnets()

actor.set_train()

init = tf.global_variables_initializer()
saver = tf.train.Saver()
print('Starting Training')
with tf.Session() as sess:
	init.run()
	#actor.set_session(sess)
	actorAsync.set_session(sess)
	criticAsync.set_session(sess)

	if restore_model == True:
		#saver.restore(sess, "./models/pretrain3_2.ckpt")
		saver.restore(sess, "./models/train3_1.ckpt")
		print("Model restored.")
	else:
		for epoch in range(num_epochs+1):

			bbb, epoch_cost = actor.pretrain(X_train.T, Y_train.T)
	        
	        # Print the cost every 100 epochs
	        if epoch % 100 == 0:
	            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
	        if epoch % 5 == 0:
	            costs.append(epoch_cost)

		actorAsync.async_update(tau=1) # copy trained actor net to actor target
		save_path = saver.save(sess, "./models/pretrain3_2.ckpt")
		print("Model saved in path: %s" % save_path)
	
	# evaluate costs
	cost_train = tf.losses.mean_squared_error(Y_train.T,actor.predict_action(X_train.T))
	train_loss = sess.run(cost_train)
	print('Training Loss', train_loss)

	pred_dev = actor.predict_action(X_dev.T)
	cost_dev = tf.losses.mean_squared_error(Yh_dev.T,pred_dev)
	dev_loss = sess.run(cost_dev)
	print('Dev loss', dev_loss)

	pred_test = actor.predict_action(X_test.T)
	cost_test = tf.losses.mean_squared_error(Yh_test.T,pred_test)
	test_loss = sess.run(cost_test)
	print('Test loss', test_loss)

	pred_test = actor_target.predict_action(X_test.T)
	cost_test = tf.losses.mean_squared_error(Yh_test.T,pred_test)
	test_loss = sess.run(cost_test)
	print('Test loss target', test_loss)

	np.savez('Pretrain_Predictionsf_2', pred_dev=pred_dev, pred_test=pred_test, costs=costs)

# DDPG code
"""
max_episode = 150*100
gamma = .9
tau = 0.001
memory_size = 10000
batch_size = 30 #256
memory_warmup = batch_size*3
max_explore_eps = 150*90 #100

tf.reset_default_graph()
actorAsync = AsyncNets('Actor',3)
actor,actor_target = actorAsync.get_subnets()
criticAsync = AsyncNets('Critic',3)
critic,critic_target = criticAsync.get_subnets()

init = tf.global_variables_initializer()
saver = tf.train.Saver()


"""