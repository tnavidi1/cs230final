# Data Preprocessing

from scipy.io import loadmat
import numpy as np
from powerflowEnv import *

# load data
nodesStorage = 3 # case with only 1 storage node
Vmin = .95 # max and min voltage
Vmax = 1.05
DataDict0 = loadmat('DemoData.mat')
rDemandFull = np.matrix(DataDict0['rDemand'])
# Switch transformer and commercial building demand
rDemandFull[2,:] = rDemandFull[1,:]
rDemandFull[1,:] = rDemandFull[1,:] - rDemandFull[1,:]
ppc = DemoCase7()
Ybus = GetYbus(ppc)

DataDict = np.load('DemoRLdata_1storage.npz')
#DataDict = np.load('DemoRLdata_1storage_NoControl.npz')

netDemandFull = DataDict['netDemandFull']
umax = DataDict['umax']
Uall_opt = DataDict['Uall'][1,:]
#Uall_disc, bins = pd.cut(Uall_opt, 5, labels=False, retbins=True)
vVios_opt = DataDict['vVios']

#print('u_opt',Uall_opt[0:30]/umax[1])
u_mean = np.mean(Uall_opt.reshape([24,3600/24], order='F')/umax[1], axis=1)
u_mean[4] += .016
u_mean[15] += .331
u_mean[21:24] = 0
qmax = 3


# Load Prices TOU
prices = np.matrix(np.hstack((.25*np.ones((1,16)) , .35*np.ones((1,5)), .25*np.ones((1,3)))))
#umax = umax*3/4 # make maximum charging weaker take 4 timesteps for full
#umin = -umax

# Normalize state space
netDemandFull = netDemandFull - np.mean(netDemandFull, axis=1, keepdims=True)
netDemandFull = netDemandFull/np.std(netDemandFull, axis=1, keepdims=True)

X = netDemandFull[2:,:]
Nnodes, horizon = X.shape
t_state = 24
X = X.reshape([Nnodes*t_state,horizon/t_state], order='F') # order of vector is all nodes then next time

X_train = np.tile(X, (1,100))
X_train = X_train + np.random.randn(*X_train.shape)*.2
X_train = X_train/np.std(X_train, axis=1, keepdims=True)

Y_train = np.tile(u_mean.reshape(24,1), (1,X_train.shape[1]))
print('Y_train shape',Y_train.shape)
print('X train shape',X_train.shape)

X_dev = X[:,0:X.shape[1]/2]
X_test = X[:,X.shape[1]/2:]
print('X dev shape',X_dev.shape)
print('X test shape',X_test.shape)

Ureal = Uall_opt.reshape([24,3600/24], order='F')/umax[1]

Y_dev = Ureal[:,0:Ureal.shape[1]/2]
Y_test = Ureal[:,Ureal.shape[1]/2:]
print('Y dev shape',Y_dev.shape)
print('Y test shape',Y_test.shape)

#np.savez('Data_Processed', X_train=X_train, Y_train=Y_train, X_dev=X_dev, Y_dev=Y_dev, X_test=X_test, Y_test=Y_test)
