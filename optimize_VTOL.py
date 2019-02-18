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

def motor_cnstr(pids):
    kp_z = pids[0]
    ki_z = pids[1]
    kd_z = pids[2]
    kp_h = pids[3]
    ki_h = pids[4]
    kd_h = pids[5]
    kp_th = pids[6]
    kd_th = pids[7]
    state_hist, uu_hist = sim.simulate(kp_z, ki_z, kd_z,
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

nonlcon = []

nonlcon.append(scipy.optimize.NonlinearConstraint(Param.rt_ratio,
                                                  Param.lb_tr_ratio, Param.ub_tr_ratio,
                                                  grad(Param.rt_ratio)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(Param.zeta_lon,
                                                  Param.lb_zeta, Param.ub_zeta,
                                                  grad(Param.zeta_lon)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(Param.zeta_lat,
                                                  Param.lb_zeta, Param.ub_zeta,
                                                  grad(Param.zeta_lat)))
#
nonlcon.append(scipy.optimize.NonlinearConstraint(Param.zeta_th,
                                                  Param.lb_zeta, Param.ub_zeta,
                                                  grad(Param.zeta_th)))
#
# ======================================
# constrain each motor thrust, theta,
nonlcon.append(scipy.optimize.NonlinearConstraint(motor_cnstr,
                                                  Param.lb_motor, Param.ub_motor,
                                                  jacobian(motor_cnstr)))
#



# ======================================
# objective function
# we want the aircraft to behave like a first order system
#
def obj_fun( pids ):
    kp_z = pids[0]
    ki_z = pids[1]
    kd_z = pids[2]
    kp_h = pids[3]
    ki_h = pids[4]
    kd_h = pids[5]
    kp_th = pids[6]
    kd_th = pids[7]
    state_hist, uu_hist = sim.simulate( kp_z, ki_z, kd_z,
                                        kp_h, ki_h, kd_h,
                                        kp_th, kd_th)
    #
    cost = np.linalg.norm(state_hist[:2,:] - Param.target)
    return cost
#
# ======================================
x0 = np.array([Param.kp_z, Param.ki_z, Param.kd_z,
               Param.kp_h, Param.ki_h, Param.kd_h,
               Param.kp_th, Param.kd_th])
hess = scipy.optimize.BFGS(exception_strategy='skip_update')
result = scipy.optimize.minimize(obj_fun, x0, jac=grad(obj_fun), hess=hess,
                                 constraints=nonlcon, callback=iteration_callback,
                                 method='trust-constr')

#
