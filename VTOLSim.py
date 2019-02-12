from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import autograd.numpy as np

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
class Gains:
    def __init__(self, kp_z, ki_z, kd_z, kp_h, ki_h, kd_h, kp_th, kd_th):
        self.kp_z = kp_z
        self.ki_z = ki_z
        self.kd_z = kd_z
        self.kp_h = kp_h
        self.ki_h = ki_h
        self.kd_h = kd_h
        self.kp_th = kp_th
        self.kd_th = kd_th

        # self.pids = kp_z
#

def set_col(my_eye, A, b, ii):
    nrows, ncols = A.shape
    select = my_eye[ii, :]
    # select = np.hstack((np.zeros(nrows,ii-1), np.ones(nrows, 1), np.zeros(nrows, ncols-ii)))
    A = A + select[None,:]*b[:, None]
    return A

def simulate(kp_z, ki_z, kd_z, kp_h, ki_h, kd_h, kp_th, kd_th):
    gains = Gains(kp_z, ki_z, kd_z, kp_h, ki_h, kd_h, kp_th, kd_th)
    VTOL = VTOLDynamics()
    ctrl = VTOLController(gains)
    z_reference = signalGenerator(amplitude=4.0, frequency=0.02)
    h_reference = signalGenerator(amplitude=3.0, frequency=0.03)

    # instantiate the simulation plots and animation
    # dataPlot = plotData()
    # animation = VTOLAnimation()
    t_span = np.arange( Param.t_start, Param.t_end + Param.Ts, Param.Ts )
    n_steps = len(t_span)
    my_eye = np.eye(n_steps)
    state_hist = np.zeros([6, n_steps])
    z_ref_hist = 5.0 + z_reference.square_batch(t_span)
    h_ref_hist = 5.0 + h_reference.square_batch(t_span)
    ref_hist = np.vstack( (z_ref_hist, h_ref_hist) )
    uu_hist = np.zeros([2, n_steps])

    for ii, tt in enumerate(t_span):
        uu = ctrl.uu(ref_hist[:,ii], VTOL.outputs())  # Calculate the control value
        VTOL.propagateDynamics(np.dot(Param.mixing, uu))  # Propagate the dynamics
        uu_hist = set_col(my_eye, uu_hist, uu, ii)
        state_hist = set_col(my_eye, state_hist, VTOL.states(), ii)
        # set_trace()
        # uu_hist[:,ii] = uu
        # state_hist[:,ii] = VTOL.states()

    return t_span, state_hist, ref_hist, uu_hist
    #
#
# set_trace()
def obj_fun( pids ):
    kp_z = pids[0]
    ki_z = pids[1]
    kd_z = pids[2]
    kp_h = pids[3]
    ki_h = pids[4]
    kd_h = pids[5]
    kp_th = pids[6]
    kd_th = pids[7]
    t_span, state_hist, ref_hist, uu_hist = simulate(kp_z, ki_z, kd_z,
                                                     kp_h, ki_h, kd_h,
                                                     kp_th, kd_th)
    #
    cost = np.linalg.norm(state_hist[:2,:] - ref_hist)
    return cost
#
def sim_and_plot():
    t_span, state_hist, ref_hist, uu_hist = simulate(Param.kp_z, Param.ki_z, Param.kd_z,
                                                     Param.kp_h, Param.ki_h, Param.kd_h,
                                                     Param.kp_th, Param.kd_th)
    #
    print("Plotting...")
    dataPlot = plotData()
    dataPlot.batchUpdatePlots(t_span, state_hist, ref_hist[0], ref_hist[1], uu_hist[0], uu_hist[1])

    # Keeps the program from closing until the user presses a button.
    print('Press key to close')
    plt.waitforbuttonpress()
    plt.close()

#
