# Network classes
# Code adapted from github user IgnacioCarlucho for DDPG implementation of Mountain Car from AI gym

import numpy as np
import tensorflow as tf
from functools import partial

class Actor(object):

    def __init__(self, n_x, n_y, layers, reg_scale=0.1, name='actor_net',pretrain=False):
    	"""
    	Inputs: input and output dimensions and list of layers dimensions not including input/output
    			reg_scale is scaling for l2 regularizer
    	"""
        self.n_x = n_x # number of input variables
        self.n_y = n_y # number of output variables
        self.name = name
        self.sess = None
        self.build_model(layers, reg_scale)
        if pretrain==False:
        	self.build_train()

    def build_model(self, layers, reg_scale):
        activation = tf.nn.elu # use exponential linear unit as activation
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer() # constant variance initializer
        kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale) # add l2 regualrization
        default_dense = partial(tf.layers.dense,\
                                activation=activation,\
                                kernel_initializer=kernel_initializer,\
                                kernel_regularizer=kernel_regularizer)
        with tf.variable_scope(self.name) as scope:
        	# Make network
            observation = tf.placeholder(tf.float32,shape=[None,self.n_x])
            
            hid1 = default_dense(observation,layers[0])
            hid2 = default_dense(hid1,layers[1])
            
            action = default_dense(hid2,self.n_y,activation=tf.nn.tanh,use_bias=False) # tanh activation for output since charging between -1 and 1
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
        self.observation,self.action,self.trainable_vars = observation,action,trainable_vars

    def set_train(self, learning_rate=.0001):
    	with tf.variable_scope(self.name) as scope:
    	    #Define cost and optimizer
    	    labels = tf.placeholder(tf.float32, [None,self.n_y])
            cost = tf.losses.mean_squared_error(labels,self.action) # cost is mean squared error
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        self.labels, self.cost, self.optimizer = labels, cost, optimizer

    def pretrain(self, obs_batch, labels_batch):
    	#self.optimizer.run(session=self.sess,feed_dict={self.observation:obs_batch, self.labels:labels_batch})
    	return self.sess.run([self.optimizer, self.cost], feed_dict={self.observation:obs_batch, self.labels:labels_batch})

    def predict_action(self,obs_batch):
        return self.action.eval(session=self.sess,feed_dict={self.observation:obs_batch})

    def set_session(self,sess):
        self.sess = sess
    
    
    #DDPG code
    def build_train(self,learning_rate = 0.0001):
        with tf.variable_scope(self.name) as scope:
            action_grads = tf.placeholder(tf.float32,[None,self.n_y])
            var_grads = tf.gradients(self.action,self.trainable_vars,-action_grads)
            train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(var_grads,self.trainable_vars))
        self.action_grads,self.train_op = action_grads,train_op

    def train(self,obs_batch,action_grads):
        batch_size = len(action_grads)
        self.train_op.run(session=self.sess,feed_dict={self.observation:obs_batch,self.action_grads:action_grads/batch_size})

    def get_trainable_dict(self):
        return {var.name[len(self.name):]: var for var in self.trainable_vars}

class Critic(object):
    def __init__(self, n_x, n_y, layers, reg_scale=0.1, name='critic_net',pretrain=False):
        self.n_observation = n_x
        self.n_action = n_y
        self.name = name
        self.sess = None
        self.build_model(layers, reg_scale)
        self.build_train()
        
    def build_model(self, layers, reg_scale):
        activation = tf.nn.elu
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale)
        default_dense = partial(tf.layers.dense,\
                                activation=activation,\
                                kernel_initializer=kernel_initializer,\
                                kernel_regularizer=kernel_regularizer)
        with tf.variable_scope(self.name) as scope:
            observation = tf.placeholder(tf.float32,shape=[None,self.n_observation])
            action = tf.placeholder(tf.float32,shape=[None,self.n_action])
            hid1 = default_dense(observation,layers[0]) #32
            hid2 = default_dense(action,layers[1]) #32
            hid3 = tf.concat([hid1,hid2],axis=1)
            hid4 = default_dense(hid3,layers[2]) #128
            Q = default_dense(hid4,layers[3], activation=None) #1
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
        self.observation,self.action,self.Q,self.trainable_vars= observation,action,Q,trainable_vars
    
    def build_train(self,learning_rate=0.001):
        with tf.variable_scope(self.name) as scope:
            Qexpected = tf.placeholder(tf.float32,shape=[None,1])
            loss = tf.losses.mean_squared_error(Qexpected,self.Q)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)
        self.Qexpected,self.train_op = Qexpected,train_op
        self.action_grads = tf.gradients(self.Q,self.action)[0]
    
    def predict_Q(self,obs_batch,action_batch):
        return self.Q.eval(session=self.sess,\
                           feed_dict={self.observation:obs_batch,self.action:action_batch})
    
    def compute_action_grads(self,obs_batch,action_batch):
        return self.action_grads.eval(session=self.sess,\
                               feed_dict={self.observation:obs_batch,self.action:action_batch})
    def train(self,obs_batch,action_batch,Qexpected_batch):
        self.train_op.run(session=self.sess,\
                          feed_dict={self.observation:obs_batch,self.action:action_batch,self.Qexpected:Qexpected_batch})
    
    def set_session(self,sess):
        self.sess = sess
        
    def get_trainable_dict(self):
        return {var.name[len(self.name):]: var for var in self.trainable_vars}


class AsyncNets(object):
    def __init__(self, n_x, n_y, layers, reg_scale=0.1, class_name='Actor',pretrain=False):
        class_ = eval(class_name)
        self.net = class_(n_x, n_y, layers, name=class_name,pretrain=pretrain) # create net
        self.target_net = class_(n_x, n_y, layers, name='{}_target'.format(class_name)) # create target net
        self.TAU = tf.placeholder(tf.float32,shape=None)
        self.sess = None
        self.__build_async_assign()
    
    def __build_async_assign(self):
        net_dict = self.net.get_trainable_dict()
        target_net_dict = self.target_net.get_trainable_dict()
        keys = net_dict.keys()
        async_update_op = [target_net_dict[key].assign((1-self.TAU)*target_net_dict[key]+self.TAU*net_dict[key]) \
                           for key in keys]
        self.async_update_op = async_update_op
    
    def async_update(self,tau=0.01):
        self.sess.run(self.async_update_op,feed_dict={self.TAU:tau})
    
    def set_session(self,sess):
        self.sess = sess
        self.net.set_session(sess)
        self.target_net.set_session(sess)
    
    def get_subnets(self):
        return self.net, self.target_net

def UONoise(n_action): # random normal noise with momentum
    theta = 0.15
    sigma = 0.2
    state = np.zeros(n_action)
    while True:
        yield state
        state += -theta*state+sigma*np.random.randn(n_action)

#totally random noise not with momentum
def NNoise(n_action):
    sigma = 0.5
    state = 0
    while True:
        yield state
        state = sigma*np.random.randn(n_action)

from collections import deque
class Memory(object):
    def __init__(self,memory_size=10000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        
    def __len__(self):
        return len(self.memory)
    
    def append(self,item):
        self.memory.append(item)
        
    def sample_batch(self,batch_size=256):
        idx = np.random.permutation(len(self.memory))[:batch_size]
        return [self.memory[i] for i in idx]







