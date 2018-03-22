# main DDPG script
# Code adapted from github user IgnacioCarlucho for DDPG implementation of Mountain Car from AI gym

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from DDPGModels import *
from powerflowEnv import *

import time

start_time = time.time()

restore_model = True

# Loading Initial Training Data
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
X_obs = np.hstack((X_dev,X_test))

DataDict = loadmat('DemoData.mat')
rDemandFull = np.matrix(DataDict['rDemand'])
# Switch transformer and commercial building demand
rDemandFull[2,:] = rDemandFull[1,:]
rDemandFull[1,:] = rDemandFull[1,:] - rDemandFull[1,:]
ppc = DemoCase7()
nodesStorage = 3 # case with only 1 storage node
Vmin = .95 # max and min voltage
Vmax = 1.05

DataDict = np.load('DemoRLdata_1storage.npz')
netDemandFull = DataDict['netDemandFull']
umax = DataDict['umax']
umax = umax[1] # extract just the charge of the one storage node
Nnodes, horizon = netDemandFull.shape

# Load Prices TOU
prices = np.matrix(np.hstack((.25*np.ones((1,16)) , .35*np.ones((1,5)), .25*np.ones((1,3)))))

# size of state and action
n_x = X_train.shape[0] # 5 nodes * 24 hours
n_y = Y_train.shape[0] # 1 control node * 24 hours

# size of 2 hidden layers in network
layers_actor = [100,50] # hidden units for state, layer2
layers_critic = [100,50,25,1] #hidden units for state,action,layer2 combine state and action, output

tf.reset_default_graph()

actorAsync = AsyncNets(n_x, n_y, layers_actor, class_name='Actor')
actor,actor_target = actorAsync.get_subnets()
criticAsync = AsyncNets(n_x, n_y, layers_critic, class_name='Critic')
critic,critic_target = criticAsync.get_subnets()

max_episode = 40500+1 #200
gamma = 0.99
tau = 0.001
memory_size = 10000 #10000
batch_size = 256 #256
memory_warmup = batch_size*3 # *3
max_explore_eps = 30000 #100
Q_train_eps = 20900 #2000
heuristic_eps = 900 #1000
# first just train Q on heuristic
# next just train Q on random actions
# next train Q and A on random actions
# last train Q and A on deterministic actions


VioDays = []
VioAmount = []
all_scores = []
max_score = -100
u_all = np.zeros((150,24))

end_init = time.time()
print('initialization time',end_init-start_time)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
print('Creating Networks')

with tf.Session() as sess:
	init.run()
	actorAsync.set_session(sess)
	criticAsync.set_session(sess)

	if restore_model == True:
		saver.restore(sess, "./models/pretrain3_2.ckpt")
		#saver.restore(sess, "./models/train3_2.ckpt")
		print("Model restored.")

	iteration = 0
	episode = 0
	episode_score = 0
	episode_steps = 0
	vio_total = 0
	noise = NNoise(n_y)
	memory = Memory(memory_size)

	while episode < max_episode:

		e_start = time.time()

		d = episode % (horizon/24)

		obs = np.reshape(X_obs[:,d],[1,n_x])

		#print('\riter {}, ep {}'.format(iteration,episode),end='')
		action = actor.predict_action(obs)

		if (episode-heuristic_eps)<max_explore_eps and (episode-heuristic_eps)>=0: # exploration policy
			p = (episode-heuristic_eps)/max_explore_eps
			action = action*p + (1-p)*next(noise)
		
		action = clip_u(action)
		reward, vio = PF_Sim_Out_Day(ppc, netDemandFull[:, d*24:(d+1)*24], rDemandFull[:, d*24:(d+1)*24], nodesStorage, umax*action, umax, Vmin, Vmax)

		"""
		if reward < 0:
			print('Vio on day',d)
			VioDays.append(d)
			VioAmount.append(vio)
		"""

		reward += -.5*np.array(np.dot(prices, action.T*umax))[0,0] # add reward for arbitrage

		# Every state is terminal
		next_obs = 0
		done = 1

		u_all[d,:] = action*umax

		#next_obs, reward, done,info = env.step(action)
		memory.append([obs[0],action[0],reward,next_obs,done])

		if iteration >= memory_warmup:
			memory_batch = memory.sample_batch(batch_size)
			extract_mem = lambda k : np.array([item[k] for item in memory_batch])
			obs_batch = extract_mem(0)
			action_batch = extract_mem(1)
			reward_batch = extract_mem(2)
			next_obs_batch = extract_mem(3)
			done_batch = extract_mem(4)

			# Ignore since every state is terminal
			#action_next = actor_target.predict_action(next_obs_batch)
			#Q_next = critic_target.predict_Q(next_obs_batch,action_next)[:,0]
			Q_next = 0

			Qexpected_batch = reward_batch + gamma*(1-done_batch)*Q_next # target Q value
			Qexpected_batch = np.reshape(Qexpected_batch,[-1,1])

			# train critic
			critic.train(obs_batch,action_batch,Qexpected_batch)
			criticAsync.async_update(tau)
			# train actor
			if episode >= Q_train_eps:
				action_grads = critic.compute_action_grads(obs_batch,action_batch)
				actor.train(obs_batch,action_grads)
				actorAsync.async_update(tau)

		episode_score += reward
		vio_total += vio
		episode_steps += 1
		iteration += 1

		if d == 149:
			print('episode', episode)
			print('score', episode_score)
			all_scores.append(episode_score)
			ARB = np.dot(prices, np.sum(u_all, axis=0))
			print('ARB', ARB)
			print('vio', vio_total)
			print('sample u', u_all[140,:])
			if episode_score > max_score:
				max_score = episode_score
				episode_of_max = episode
				max_vio = vio_total
				max_ARB = ARB
				np.savez('maxScore3_6', max_score=max_score, episode_of_max=episode_of_max, max_vio=max_vio, max_ARB=max_ARB, u_all=u_all)
				print('Hit max score and saved',max_score)
			episode += 1
			episode_score = 0
			episode_steps = 0
			vio_total = 0
			#noise = UONoise(n_y)
			#if episode%25==0:
			saver.save(sess,"./models/train3_6.ckpt")
			#np.savez('VioData3_4', VioDays=VioDays, VioAmount=VioAmount, ARB=ARB, u_all=u_all)
			np.savez('Data3_6', u_all=u_all, all_scores=all_scores)
			VioDays = []
			VioAmount = []
		else:
			episode += 1
		    #obs = next_obs



"""
	pred_test = actor.predict_action(X_test.T)
	cost_test = tf.losses.mean_squared_error(Yh_test.T,pred_test)
	test_loss = sess.run(cost_test)
	print('Test loss', test_loss)

	pred_test = actor_target.predict_action(X_test.T)
	cost_test = tf.losses.mean_squared_error(Yh_test.T,pred_test)
	test_loss = sess.run(cost_test)
	print('Test loss target', test_loss)
"""


