import numpy as np

from importlib import reload

import VTOLParam as Param
import VTOLParamHW10 as P10
import PIDControl
reload(Param)
reload(P10)
reload(PIDControl)

from PIDControl import PIDControl


class VTOLController:
    '''
        This class inherits other controllers in order to organize multiple controllers.
    '''

    def __init__(self):
        self.zCtrl = PIDControl(P10.kp_z, P10.ki_z, P10.kd_z, Param.fmax, Param.beta, Param.Ts)
        self.hCtrl = PIDControl(P10.kp_h, P10.ki_h, P10.kd_h, Param.fmax, Param.beta, Param.Ts)
        self.thetaCtrl = PIDControl(P10.kp_th, 0.0, P10.kd_th, Param.fmax, Param.beta, Param.Ts)

    def u(self, r, y):
        z_r = float(r[0])
        h_r = float(r[1])
        z = y[0]
        h = y[1]
        theta = y[2]
        F_tilde = self.hCtrl.PID(h_r, h, error_limit=1.0, flag=False)
        F = F_tilde + Param.Fe
        theta_ref = self.zCtrl.PID(z_r, z, flag=False)
        tau = self.thetaCtrl.PID(theta_ref, theta, flag=False)
        return np.array([F, tau])
