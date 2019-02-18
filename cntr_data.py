from IPython.core.debugger import set_trace
from importlib import reload

import scipy.io
import numpy as np

# import optimize_VTOL as opv
import VTOLSim as vsim
import VTOLParam as Param
reload(vsim)
reload(Param)

kp_opted = -0.0509283789
kd_opted = -0.0807472303


kp_lb = 4 * kp_opted  # Param.kp_z_up
kp_ub = -0.0001 # Param.kp_z_low
inc_kp = 100

kd_lb = 4 * kd_opted # Param.kd_z_up
kd_ub = -0.0001 # Param.kd_z_low
inc_kd = 100

kp_array = np.linspace(kp_lb, kp_ub, inc_kp)
kd_array = np.linspace(kd_lb, kd_ub, inc_kd)

# kp_z = pids[0]
# ki_z = pids[1]
# kd_z = pids[2]
# kp_h = pids[3]
# ki_h = pids[4]
# kd_h = pids[5]
# kp_th = pids[6]
# kd_th = pids[7]
pids = np.zeros(8)

pids[0] = Param.perf_kp_z
pids[1] = Param.perf_ki_z
pids[2] = Param.perf_kd_z
pids[3] = Param.perf_kp_h
pids[4] = Param.perf_ki_h
pids[5] = Param.perf_kd_h
pids[6] = Param.perf_kp_th
pids[7] = Param.perf_kd_th


rt_ratio_grid   = np.zeros( (inc_kp,inc_kd) )
zeta_lon_grid   = np.zeros( (inc_kp,inc_kd) )
zeta_lat_grid   = np.zeros( (inc_kp,inc_kd) )
zeta_th_grid    = np.zeros( (inc_kp,inc_kd) )
mot_grid        = np.zeros( (inc_kp,inc_kd) )
costs           = np.zeros( (inc_kp,inc_kd) )

# for ii, kp_z in enumerate( arange(kp_lb, kp_ub+inc_kp, inc_kp) ):
#     for jj, kd_z in enumerate( arange(kd_lb, kd_ub+inc_kd, inc_kd) ):
for ii, kp_z in enumerate( kp_array ):
    for jj, kd_z in enumerate( kd_array ):
        pids[0] = kp_z
        pids[2] = kd_z

        state_hist, mot_grid[ii,jj] = vsim.cntr_sim(kp_z,kd_z)

        rt_ratio_grid[ii,jj]    = Param.rt_ratio(pids)
        zeta_lon_grid[ii,jj]    = Param.zeta_lon(pids)
        zeta_lat_grid[ii,jj]    = Param.zeta_lat(pids)
        zeta_th_grid[ii,jj]     = Param.zeta_th(pids)
        # mot_grid        = motor_cnstr(pids)
        costs[ii,jj] = np.linalg.norm( state_hist[:2,:] - Param.target )
        #
    #
#
scipy.io.savemat('opt_costs_grid.mat',{'costs':costs,
                                        'kp_array':kp_array,
                                        'kd_array':kd_array,
                                        'rt_ratio_grid':rt_ratio_grid,
                                        'zeta_lon_grid':zeta_lon_grid,
                                        'zeta_lat_grid':zeta_lat_grid,
                                        'zeta_th_grid':zeta_th_grid,
                                        'mot_grid':mot_grid})
#

#
