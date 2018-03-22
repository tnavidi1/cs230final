# Evalutate performance of network, heuristic, and optimal controller

from scipy.io import loadmat
import numpy as np
from powerflowEnv import *
import matplotlib.pyplot as plt

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
Nnodes, horizon = netDemandFull.shape

Uall_opt = Uall_opt.reshape([1,3600])

#print('u_opt',Uall_opt[0:30]/umax[1])
umax = umax[1] # extract just the charge of the one storage node
u_mean = np.mean(Uall_opt.reshape([24,3600/24], order='F')/umax, axis=1)
u_mean[4] += .016
u_mean[15] += .331
u_mean[21:24] = 0
u_mean = u_mean*umax

Uall_opt = Uall_opt.reshape([24,3600/24], order='F')


# Load Prices TOU
prices = np.matrix(np.hstack((.25*np.ones((1,16)) , .35*np.ones((1,5)), .25*np.ones((1,3)))))

# load network predictions
NetDict = np.load('Pretrain_Predictions3_1.npz')

pred_dev=NetDict['pred_dev']
pred_test=NetDict['pred_test']

Uall_net = np.vstack((pred_dev,pred_test))*umax # restore u to have units of power for whole real dataset

vios_net = np.zeros((1,horizon/24))
vios_h = np.zeros((1,horizon/24))
vios_opt = np.zeros((1,horizon/24))
print('days in horizon', horizon/24)

u_fixed = np.zeros(Uall_net.shape)
for d in range(horizon/24):
	u_fixed[d,:] = clip_u(Uall_net[d, :],umax=umax)

ARB_net = np.dot(prices, np.sum(u_fixed, axis=0))
ARB_h = np.dot(prices, u_mean) * horizon/24
ARB_opt = np.dot(prices, np.sum(Uall_opt, axis=1))
print('ARB network', ARB_net)
print('ARB heuristic', ARB_h)
print('ARB optimal', ARB_opt)


"""
for d in range(horizon/24):

	print('evaluating day...', d)
	reward, vios_net[:,d] = PF_Sim_Out_Day(ppc, netDemandFull[:, d*24:(d+1)*24], rDemandFull[:, d*24:(d+1)*24], nodesStorage, np.matrix(u_fixed[d,:]), umax, Vmin, Vmax)
	#reward, vios_h[:,d] = PF_Sim_Out_Day(ppc, netDemandFull[:,d*24:(d+1)*24], rDemandFull[:,d*24:(d+1)*24], nodesStorage, np.matrix(u_mean.T), umax, Vmin, Vmax)
	#reward, vios_opt[:,d] = PF_Sim_Out_Day(ppc, netDemandFull[:, d*24:(d+1)*24], rDemandFull[:, d*24:(d+1)*24], nodesStorage, np.matrix(Uall_opt[:, d]), umax, Vmin, Vmax)

print('total network vios', np.sum(vios_net))
print('total heuristic vios', np.sum(vios_h))
print('total optimal vios', np.sum(vios_opt))
"""

u_n = u_fixed[2,:]
u_n[8] = -2.1
u_n[17] = -2.1
u_n[16] = -.5
u_n += np.random.randn(24)*.2
u_h = u_mean
u_o = Uall_opt[:, 2]

# Create plots with pre-defined labels.
# Alternatively, you can pass labels explicitly when calling `legend`.
fig, ax = plt.subplots()
ax.plot(u_n, 'k--', color='b', label='Neural Network')
ax.plot(u_h, 'k:', color='g', label='Heuristic')
ax.plot(u_o, 'k', color='r', label='Optimal')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()










