import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')  # add parent directory
import VTOLParam as P
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
animation = VTOLAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Get referenced inputs from signal generators
    z_ref = 5.0 + z_reference.square(t)[0]
    h_ref = 5.0 + h_reference.square(t)[0]
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot: # updates control and dynamics at faster simulation rate
        ref = np.array([[z_ref], [h_ref]])
        u = ctrl.u(ref, VTOL.outputs())  # Calculate the control value
        VTOL.propagateDynamics(P.mixing@u)  # Propagate the dynamics
        t = t + P.Ts  # advance time by Ts
    # update animation and data plots
    animation.drawVTOL(VTOL.states(), z_ref)
    dataPlot.updatePlots(t, VTOL.states(), z_ref, h_ref, u[0], u[1])
    plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
