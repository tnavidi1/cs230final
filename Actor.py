# Actor network class

import numpy as np
import tensorflow as tf
from functools import partial

class Actor(object):

    def __init__(self, n_x, n_y, layers, reg_scale=0.1, name='actor_net'):
    	"""
    	Inputs: input and output dimensions and list of layers dimensions not including input/output
    			reg_scale is scaling for l2 regularizer
    	"""
        self.n_x = n_x # number of input variables
        self.n_y = n_y # number of output variables
        self.name = name
        self.sess = None
        self.build_model(layers, reg_scale)
        #self.build_train()

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
    
    """
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

    """

"""
def initialize_parameters(layers):
	#Input: array of layer sizes including input layer
	#Output: dictionary of parameters initilzed using Xavier initialization

	parameters = {}
       
    for i in range(len(layers)-1):
	    W = tf.get_variable("W"+str(i+1), [layers[i+1],layers[i]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	    b = tf.get_variable("b"+str(i+1), [layers[i+1],1], initializer = tf.zeros_initializer())

    	parameters['W' +str(i+1)] = W
    	parameters['b' +str(i+1)] = b

    return parameters

"""