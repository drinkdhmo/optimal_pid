# VTOL Parameter File
import autograd.numpy as np
# import control as cnt

# Physical parameters of the  VTOL known to the controller
mc = 1.0  # kg
mr = 0.25  # kg
Jc = 0.0042  # kg m^2
arm = 0.3  # m
mu = 0.1  # kg/s
gravity = 9.81  # m/s^2
F_wind = 0.1  # wind disturbance force

# parameters for animation
length = 10.0

# Initial Conditions
z0 = 0.0  # initial lateral position
h0 = 0.0  # initial altitude
theta0 = 0  # initial roll angle
zdot0 = 0  # initial lateral velocity
hdot0 = 0  # initial climb rate
thetadot0 = 0  # initial roll rate
target0 = 0

# Simulation Parameters
t_start = 0.0  # Start time of simulation
t_end = 1.0  # End time of simulation
Ts = 0.01  # sample time for simulation
t_plot = 0.1  # the plotting and animation is updated at this rate

# saturation limits
fmax = 10.0  # Max Force, N

# dirty derivative parameters
sigma = 0.05  # cutoff freq for dirty derivative
beta = (2.0*sigma-Ts)/(2.0*sigma+Ts)  # dirty derivative gain

# equilibrium force
Fe = (mc+2.0*mr)*gravity

# mixing matrix
mixing = np.linalg.inv(np.array([[1.0, 1.0],
                                  [arm, -arm]]))
#

# ======================================
# ======================================
# ======================================


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
kp_h = Delta_cl_d[2]*(mc+2.0*mr)  # kp - altitude
kd_h = Delta_cl_d[1]*(mc+2.0*mr)  # kd = altitude
Fe = (mc+2.0*mr)*gravity  # equilibrium force

# PD gains for lateral inner loop
b0       = 1.0/(Jc+2.0*mr*arm**2)
tr_th    = tr_z/M
wn_th    = 2.2/tr_th
kp_th  = wn_th**2.0/b0
kd_th  = 2.0*zeta_th*wn_th/b0

#PD gain for lateral outer loop
b1       = -Fe/(mc+2.0*mr)
a1       = mu/(mc+2.0*mr)
wn_z     = 2.2/tr_z
kp_z   = wn_z**2.0/b1
kd_z   = (2.0*zeta_z*wn_z-a1)/b1

# print('kp_z: ', kp_z)
# print('ki_z: ', ki_z)
# print('kd_z: ', kd_z)
# print('kp_h: ', kp_h)
# print('ki_z: ', ki_h)
# print('kd_h: ', kd_h)
# print('kp_th: ', kp_th)
# print('kd_th: ', kd_th)

#
