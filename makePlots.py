# load file of max performance

import numpy as np
import matplotlib.pyplot as plt

DataDict = np.load('maxScore3_4.npz')
uall = DataDict['u_all']
max_score=DataDict['max_score']
episode_of_max=DataDict['episode_of_max']
max_vio=DataDict['max_vio']
max_ARB=DataDict['max_ARB']
u_all=DataDict['u_all']

print('max_score',max_score)
print(max_vio)
print(max_ARB)

NetDict = np.load('Pretrain_Predictionsf_1.npz')

pred_dev=NetDict['pred_dev']
pred_test=NetDict['pred_test']

Uall_net = np.vstack((pred_dev,pred_test)) # restore u to have units of power for whole real dataset

print(Uall_net.shape)
print(Uall_net[0:10,:])

from powerflowEnv import *

u = clip_u(Uall_net[2,:])

plt.figure(1)
plt.plot(u)
plt.show()

# Create plots with pre-defined labels.
# Alternatively, you can pass labels explicitly when calling `legend`.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='')
ax.plot(a, d, 'k:', label='')
ax.plot(a, c+d, 'k', label='')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()


