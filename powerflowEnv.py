# powerflow environment

from pypower.api import runpf, ppoption, makeYbus
from scipy import sparse
import numpy as np

def DemoCase7():

    ## PYPOWER Case Format : Version 2
    ppc = {'version': '2'}

    ##-----  Power Flow Data  -----##
    ## system KVA base
    ppc['baseMVA'] = 1

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc['bus'] = np.array([
        [0, 3, 30, 14.5296631, 0, 0, 1, 1, 0, 12.35, 1, 1.05, 0.95],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 12.35, 1, 1.05, 0.95],
        [2, 1, 0, 0, 0, 0, 1, 1, 0, 12.35, 1, 1.05, 0.95],
        [3, 1, 0, 0, 0, 0, 1, 1, 0, 12.35, 1, 1.05, 0.95],
        [4, 1, 0, 0, 0, 0, 1, 1, 0, 12.35, 1, 1.05, 0.95],
        [5, 1, 0, 0, 0, 0, 1, 1, 0, 12.35, 1, 1.05, 0.95],
        [6, 1, 0, 0, 0, 0, 1, 1, 0, 12.35, 1, 1.05, 0.95],
    ])

    ## generator data
    # bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf
    ppc['gen'] = np.array([
        [0, 30, 12, 300, -300, 1, 1, 1, 250, -250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    ## branch data
    # fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
    ppc['branch'] = np.array([
        [0, 1, 0.00169811011, 0.00529757905, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [0, 2, 0.00443248701, 0.000603189693, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 3, 0.00301594847, 0.00103189693, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 5, 0.000603189693, 0.000203248701, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 4, 0.00040307168, 0.000101594847, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [5, 6, 0.00100295366, 0.000299940992, 0, 250, 250, 250, 0, 0, 1, -360, 360],
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc['gencost'] = np.array([
        [2, 0, 0, 2, 15, 0],
    ])

    return ppc

def PF_Sim_Out(ppc, pDemand, rDemand, nodesStorage, action, umax, Vmin, Vmax, price):
	# outer loop for recalculating PF after adjusting root voltage
	# also changes action to actual U value
	# also calculates reward from violations
	rootV2 = np.matrix(np.ones((1,1))) # no control over root voltage for first run
	U = (action - 2)*umax/2
	runVoltage = PF_Sim(ppc, pDemand, rDemand, nodesStorage, U, rootV2)
	maxV = np.max(runVoltage)
	minV = np.amin(runVoltage)
	maxmin = np.vstack((maxV,minV))
	rootV2 = 2 - np.mean(maxmin,0)
	rootV2 = np.matrix(np.square(rootV2))
	runVoltage = PF_Sim(ppc, pDemand, rDemand, nodesStorage, U, rootV2)

	# calculate reward from violations and charging
	vGreater = (runVoltage-Vmax).clip(min=0)
	vLess = (Vmin-runVoltage).clip(min=0)

	vioTotal = np.sum(np.square(vGreater+vLess))
	vReward = -10000*vioTotal # scale reward of violations by 10000

	return vReward, vioTotal

def PF_Sim_Out_cont(ppc, pDemand, rDemand, nodesStorage, action, umax, Vmin, Vmax, price):
	# outer loop for recalculating PF after adjusting root voltage
	# also changes action to actual U value
	# also calculates reward from violations
	rootV2 = np.matrix(np.ones((1,1))) # no control over root voltage for first run
	U = action*umax
	runVoltage = PF_Sim(ppc, pDemand, rDemand, nodesStorage, U, rootV2)
	maxV = np.max(runVoltage)
	minV = np.amin(runVoltage)
	maxmin = np.vstack((maxV,minV))
	rootV2 = 2 - np.mean(maxmin,0)
	rootV2 = np.matrix(np.square(rootV2))
	runVoltage = PF_Sim(ppc, pDemand, rDemand, nodesStorage, U, rootV2)

	# calculate reward from violations and charging
	vGreater = (runVoltage-Vmax).clip(min=0)
	vLess = (Vmin-runVoltage).clip(min=0)

	vioTotal = np.sum(np.square(vGreater+vLess))
	vReward = -1000*vioTotal # scale reward of violations
	#cReward = (1-abs(action)**.5)/24/8
	cReward = 0
	totalReward = np.matrix(vReward+cReward)

	return totalReward, vioTotal

def PF_Sim_Out_Day(ppc, pDemand, rDemand, nodesStorage, action, umax, Vmin, Vmax):
	# outer loop for recalculating PF after adjusting root voltage for the whole day
	# action should be actual U value and has dimensions nodes X time
	# also calculates reward from violations
	rootV2 = np.matrix(np.ones((1,1))) # no control over root voltage for first run
	U = action
	runVoltage = np.zeros((7,24))
	for i in range(24):
		runVoltage[:,i] = PF_Sim(ppc, pDemand[:,i], rDemand[:,i], nodesStorage, U[i], rootV2)
		maxV = np.max(runVoltage[:,i])
		minV = np.amin(runVoltage[:,i])
		maxmin = np.vstack((maxV,minV))
		rootV2 = 2 - np.mean(maxmin,0)
		rootV2 = np.matrix(np.square(rootV2))
		runVoltage[:,i] = PF_Sim(ppc, pDemand[:,i], rDemand[:,i], nodesStorage, U[i], rootV2)

	# calculate reward from violations and charging
	vGreater = (runVoltage-Vmax).clip(min=0)
	vLess = (Vmin-runVoltage).clip(min=0)

	vioTotal = np.sum(np.square(vGreater+vLess))
	vReward = -1000*vioTotal # scale reward of violations
	#cReward = (1-abs(action)**.5)/24/8
	cReward = 0
	totalReward = np.matrix(vReward+cReward)

	return totalReward, vioTotal

def PF_Sim(ppc, pDemand, rDemand, nodesStorage, U, rootV2):
	"""
	Uses PyPower to calculate PF to simulate node voltages after storage action
	Inputs: ppc - PyPower case dictionary
		pDemand/rDemand - true values of real and reactive power demanded
		nodesStorage - list of storage nodes indexes
		U - storage control action
		rootV2 - voltage of the substation node
	Outputs: runVoltage - (buses X time) array of voltages
	"""

	nodesNum = 7 #pDemand.shape tuple doesnt work...
	runVoltage = np.zeros((nodesNum,1))

	pLoad = np.copy(pDemand)
	pLoad[nodesStorage] = pLoad[nodesStorage] + U
	rLoad = rDemand
	rootVoltage = np.sqrt(rootV2)
	ppc['bus'][:,2] = pLoad.flatten()
	ppc['bus'][:,3] = rLoad.flatten()
	#ppc['bus'][rootIdx,7] = rootVoltage # Doesnt actually set PF root voltage
	
	ppopt = ppoption(VERBOSE = 0, OUT_ALL = 0)
	ppc_out = runpf(ppc, ppopt)

	rootVdiff = rootVoltage - 1
	runVoltage = ppc_out[0]['bus'][:,7] + rootVdiff

	return runVoltage

def GetYbus(ppc):
	Ybus, j1, j2 = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

	return Ybus
