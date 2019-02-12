import numpy as np

from importlib import reload

import VTOLParam as Param
import PIDControl
reload(Param)
reload(PIDControl)

from PIDControl import PIDControl


class VTOLController:
    '''
        This class inherits other controllers in order to organize multiple controllers.
    '''

    def __init__(self):
        self.zCtrl = PIDControl(Param.kp_z, Param.ki_z, Param.kd_z, Param.fmax, Param.beta, Param.Ts)
        self.hCtrl = PIDControl(Param.kp_h, Param.ki_h, Param.kd_h, Param.fmax, Param.beta, Param.Ts)
        self.thetaCtrl = PIDControl(Param.kp_th, 0.0, Param.kd_th, Param.fmax, Param.beta, Param.Ts)

    def uu(self, r, y):
        z_r = r[0]
        h_r = r[1]
        z = y[0]
        h = y[1]
        theta = y[2]
        F_tilde = self.hCtrl.PID(h_r, h, error_limit=1.0, flag=False)
        F = F_tilde + Param.Fe
        theta_ref = self.zCtrl.PID(z_r, z, flag=False)
        tau = self.thetaCtrl.PID(theta_ref, theta, flag=False)
        return np.array([F, tau])
