#from IPython.core.debugger import set_trace
import pdb
import autograd.numpy as np
from autograd import grad, jacobian
import scipy.optimize
import time

from importlib import reload

import VTOLParam as Param
import VTOLSim as sim

reload(Param)
reload(sim)

# kp_z = pids[0]
# ki_z = pids[1]
# kd_z = pids[2]
# kp_h = pids[3]
# ki_h = pids[4]
# kd_h = pids[5]
# kp_th = pids[6]
# kd_th = pids[7]
# set_trace()
def rt_lon_cnstr(pids):
    kp_h = pids[3]
    tr_h = 2.2 * np.sqrt( (Param.mc+2*Param.mr) / kp_h )
    return tr_h
    #
#
def rt_lat_cnstr(pids):
    kp_z = pids[0]
    # tr_z = -(2.2**2) / (Param.gravity * kp_z**2)
    tr_z = 2.2 / (-Param.gravity * kp_z)**0.5
    return tr_z
    #
#
def rt_th_cnstr(pids):
    kp_th = pids[6]
    tr_th = 2.2 * np.sqrt( (Param.Jc+2.0*Param.mr*Param.arm**2) / kp_th )
    return tr_th
    #
#
def zeta_lon_cnstr(pids):
    kd_h = pids[5]
    zeta_h = (kd_h / ( 4.4 * (Param.mc + 2*Param.mr))) * rt_lon_cnstr(pids)
    return zeta_h
    #
#
def zeta_lat_cnstr(pids):
    kd_z = pids[2]
    zeta_z = (( -Param.gravity * kd_z + ((Param.mu)/(Param.mc + 2*Param.mr)) ) / 4.4) * rt_lat_cnstr(pids)
    return zeta_z
    #
#
def zeta_th_cnstr(pids):
    kd_th = pids[7]
    zeta_th = (kd_th / ( 4.4 * (Param.Jc + 2 * Param.mr * Param.arm**2))) * rt_th_cnstr(pids)
    return zeta_th
    #
#
def motor_cnstr(pids):
    kp_z = pids[0]
    ki_z = pids[1]
    kd_z = pids[2]
    kp_h = pids[3]
    ki_h = pids[4]
    kd_h = pids[5]
    kp_th = pids[6]
    kd_th = pids[7]
    t_span, state_hist, ref_hist, uu_hist = sim.simulate(kp_z, ki_z, kd_z,
                                                         kp_h, ki_h, kd_h,
                                                         kp_th, kd_th)
    #
    motors = np.dot(Param.mixing, uu_hist)
    max_motor = np.max(motors, axis=1)
    return max_motor
    #
#
iteration_time = time.time()
def iteration_callback(xk, state):
    global iteration_time
    now = time.time()

    # pdb.set_trace()
    print("\n#############################################################")
    print("Iteration: {}".format(state.nit))
    print("cost: {}".format(state.fun))
    print("duration: {} s".format(now - iteration_time))
    print("x: {}".format(state.x))
    print("constraints: {}".format(state.constr))
    # print("success: {}".format(state.success))
    print("status: {}".format(state.status))
    # print("message: " + state.message)
    print("gradient: {}".format(state.grad))
    print("nfev: {}".format(state.nfev))
    print("njev: {}".format(state.njev))
    # print("max constraint violation: {}".format(state.maxcv))
    print("#############################################################")
    iteration_time = now
    return False
    #
#
# ======================================
# ======================================
# set_trace()
nonlcon = []
nonlcon.append(scipy.optimize.NonlinearConstraint(rt_lon_cnstr,
                                                  Param.lb_tr, Param.up_tr,
                                                  grad(rt_lon_cnstr)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(rt_lat_cnstr,
                                                  Param.lb_tr, Param.up_tr,
                                                  grad(rt_lat_cnstr)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(rt_th_cnstr,
                                                  Param.lb_tr_in, Param.up_tr_in,
                                                  grad(rt_th_cnstr)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(zeta_lon_cnstr,
                                                  Param.lb_zeta, Param.ub_zeta,
                                                  grad(zeta_lon_cnstr)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(zeta_lat_cnstr,
                                                  Param.lb_zeta, Param.ub_zeta,
                                                  grad(zeta_lat_cnstr)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(zeta_th_cnstr,
                                                  Param.lb_zeta, Param.ub_zeta,
                                                  grad(zeta_th_cnstr)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(motor_cnstr,
                                                  Param.lb_motor, Param.ub_motor,
                                                  jacobian(motor_cnstr)))
#


# ======================================
# constrain each motor thrust, theta,

#


# ======================================
x0 = np.array([Param.kp_z, Param.ki_z, Param.kd_z,
               Param.kp_h, Param.ki_h, Param.kd_h,
               Param.kp_th, Param.kd_th])
hess = scipy.optimize.BFGS(exception_strategy='skip_update')
result = scipy.optimize.minimize(sim.obj_fun, x0, jac=grad(sim.obj_fun), hess=hess,
                                 constraints=nonlcon, callback=iteration_callback,
                                 method='trust-constr')

#
