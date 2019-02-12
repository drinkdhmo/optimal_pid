from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import numpy as np

from importlib import reload

import VTOLParam as Param
import VTOLDynamics
import VTOLController
import signalGenerator
import VTOLAnimation
import plotData

reload(Param)
reload(VTOLDynamics)
reload(VTOLController)
reload(signalGenerator)
reload(VTOLAnimation)
reload(plotData)

from VTOLDynamics import VTOLDynamics
from VTOLController import VTOLController
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from plotData import plotData

# instantiate VTOL, controller, and reference classes
VTOL = VTOLDynamics()
ctrl = VTOLController()
z_reference = signalGenerator(amplitude=4.0, frequency=0.02)
h_reference = signalGenerator(amplitude=3.0, frequency=0.03)

# instantiate the simulation plots and animation
dataPlot = plotData()
# animation = VTOLAnimation()
# tt_hist = np.zeros(n_steps)
t_span = np.arange( Param.t_start, Param.t_end - Param.t_plot, Param.t_plot )
n_steps = len(t_span)
state_hist = np.zeros([6, n_steps])
z_ref_hist = np.zeros(n_steps)
h_ref_hist = np.zeros(n_steps)
uu_hist = np.zeros([2, n_steps])
ii = 0
tt = Param.t_start  # time starts at t_start

while tt < Param.t_end:  # main simulation loop
    # Get referenced inputs from signal generators
    z_ref = 5.0 + z_reference.square(tt)[0]
    h_ref = 5.0 + h_reference.square(tt)[0]
    # Propagate dynamics in between plot samples
    t_next_plot = tt + Param.t_plot
    while tt < t_next_plot: # updates control and dynamics at faster simulation rate
        ref = np.array([z_ref, h_ref])
        uu = ctrl.uu(ref, VTOL.outputs())  # Calculate the control value
        VTOL.propagateDynamics(Param.mixing@uu)  # Propagate the dynamics
        tt = tt + Param.Ts  # advance time by Ts
    # update animation and data plots
    # animation.drawVTOL(VTOL.states(), z_ref)
    # tt_hist[ii] = tt
    state_hist[:,ii] = VTOL.states()
    z_ref_hist[ii] = z_ref
    h_ref_hist[ii] = h_ref
    uu_hist[:,ii] = uu
    ii += 1
    # dataPlot.updatePlots(tt, VTOL.states(), z_ref, h_ref, uu[0], uu[1])
    # plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation
# set_trace()
dataPlot.batchUpdatePlots(t_span, state_hist, z_ref_hist, h_ref_hist, uu_hist[0], uu_hist[1])

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
