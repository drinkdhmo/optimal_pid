# VTOL Parameter File
import numpy as np
# import control as cnt
import sys
sys.path.append('..')  # add parent directory
import VTOLParam as P

# tuning parameters
tr_h = 3.0   # rise time for altitude - original
zeta_h = 0.707  # damping ratio for altitude
tr_z = 3.0  # rise time for outer lateral loop (position) - original
M = 10.0  # time separation between inner and outer lateral loops
zeta_z = 0.707  # damping ratio for outer lateral loop
zeta_th = 0.707  # damping ratio for inner lateral loop
ki_h = 0.5  # integrator on altitude
ki_z = 0.0  # integrator on position


# PD gains for longitudinal (altitude) control
wn_h = 2.2/tr_h   # natural frequency
Delta_cl_d = [1, 2*zeta_h*wn_h, wn_h**2.0]  # desired closed loop char eq
kp_h = Delta_cl_d[2]*(P.mc+2.0*P.mr)  # kp - altitude
kd_h = Delta_cl_d[1]*(P.mc+2.0*P.mr)  # kd = altitude
Fe = (P.mc+2.0*P.mr)*P.g  # equilibrium force

# PD gains for lateral inner loop
b0       = 1.0/(P.Jc+2.0*P.mr*P.d**2)
tr_th    = tr_z/M
wn_th    = 2.2/tr_th
kp_th  = wn_th**2.0/b0
kd_th  = 2.0*zeta_th*wn_th/b0

#PD gain for lateral outer loop
b1       = -P.Fe/(P.mc+2.0*P.mr)
a1       = P.mu/(P.mc+2.0*P.mr)
wn_z     = 2.2/tr_z
kp_z   = wn_z**2.0/b1
kd_z   = (2.0*zeta_z*wn_z-a1)/b1


print('kp_z: ', kp_z)
print('ki_z: ', ki_z)
print('kd_z: ', kd_z)
print('kp_h: ', kp_h)
print('ki_z: ', ki_h)
print('kd_h: ', kd_h)
print('kp_th: ', kp_th)
print('kd_th: ', kd_th)



