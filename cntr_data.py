from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np

# import optimize_VTOL as opv
import VTOLSim as vsim
import VTOLParam as Param
# reload(opv)
reload(vsim)
reload(Param)

kp_opted = -0.0509
kd_opted = -0.0807

# kp_delta = -0.0001
# kd_delta =

kp_lb = 4 * kp_opted  # Param.kp_z_up
kp_ub = -0.0001 # Param.kp_z_low
inc_kp = 100

kd_lb = 4 * kd_opted # Param.kd_z_up
kd_ub = -0.0001 # Param.kd_z_low
inc_kd = 100

kp_array = np.linspace(kp_lb, kp_ub+inc_kp, inc_kp)
kd_array = np.linspace(kd_lb, kd_ub+inc_kd, inc_kd)

costs = np.zeros( (inc_kp,inc_kd) )

# for ii, kp_z in enumerate( arange(kp_lb, kp_ub+inc_kp, inc_kp) ):
#     for jj, kd_z in enumerate( arange(kd_lb, kd_ub+inc_kd, inc_kd) ):
for ii, kp_z in enumerate( kp_array ):
    for jj, kd_z in enumerate( kd_array ):
        state_hist = vsim.cntr_sim(kp_z,kd_z)
        costs[ii,jj] = np.linalg.norm( state_hist[:2,:] - Param.target )
        #
    #
#



#
