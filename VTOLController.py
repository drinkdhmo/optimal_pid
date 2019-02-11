import numpy as np
import sys
sys.path.append('..')  # add parent directory
import VTOLParam as P
import VTOLParamHW10 as P10
from PIDControl import PIDControl

class VTOLController:
    '''
        This class inherits other controllers in order to organize multiple controllers.
    '''

    def __init__(self):
        self.zCtrl = PIDControl(P10.kp_z, P10.ki_z, P10.kd_z, P.fmax, P.beta, P.Ts)
        self.hCtrl = PIDControl(P10.kp_h, P10.ki_h, P10.kd_h, P.fmax, P.beta, P.Ts)
        self.thetaCtrl = PIDControl(P10.kp_th, 0.0, P10.kd_th, P.fmax, P.beta, P.Ts)

    def u(self, r, y):
        z_r = float(r[0])
        h_r = float(r[1])
        z = y[0]
        h = y[1]
        theta = y[2]
        F_tilde = self.hCtrl.PID(h_r, h, error_limit=1.0, flag=False)
        F = F_tilde + P.Fe
        theta_ref = self.zCtrl.PID(z_r, z, flag=False)
        tau = self.thetaCtrl.PID(theta_ref, theta, flag=False)
        return np.array([F, tau])
