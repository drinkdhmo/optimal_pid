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
t_span = np.arange( Param.t_start, Param.t_end + Param.Ts, Param.Ts )
n_steps = len(t_span)
state_hist = np.zeros([6, n_steps])
z_ref_hist = 5.0 + z_reference.square_batch(t_span)
h_ref_hist = 5.0 + h_reference.square_batch(t_span)
ref_hist = np.vstack( (z_ref_hist, h_ref_hist) )
uu_hist = np.zeros([2, n_steps])

for ii, tt in enumerate(t_span):
    uu = ctrl.uu(ref_hist[:,ii], VTOL.outputs())  # Calculate the control value
    VTOL.propagateDynamics(Param.mixing@uu)  # Propagate the dynamics
    uu_hist[:,ii] = uu
    state_hist[:,ii] = VTOL.states()
    #
#
# set_trace()
print("Plotting...")
dataPlot.batchUpdatePlots(t_span, state_hist, z_ref_hist, h_ref_hist, uu_hist[0], uu_hist[1])

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()

#
