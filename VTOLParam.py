# VTOL Parameter File
import autograd.numpy as np

from importlib import reload
import signalGenerator
reload(signalGenerator)
from signalGenerator import signalGenerator

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
t_end = 15.0  # End time of simulation
Ts = 0.02 # 0.01  # sample time for simulation
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
# simulation parameters
z_step = 4.0
h_step = 3.0
ref_tau = 1.0
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

perf_kp_z = -0.0509283789 # kp_z
perf_ki_z = 0.00167443749 # ki_z
perf_kd_z = -0.0807472303 # kd_z
perf_kp_h = 1.08487657 # kp_h
perf_ki_h = 0.280319924 # ki_h
perf_kd_h = 1.55297213 # kd_h
perf_kp_th = 3.84914829 # kp_th
perf_kd_th = 0.596620813 # kd_th

t_span = np.arange( t_start, t_end + Ts, Ts )
n_steps = len(t_span)
my_eye = np.eye(n_steps)
target = np.array([[5+z_step], [5+h_step]])*(1 - np.exp(-t_span/ref_tau))

z_reference = signalGenerator(amplitude=z_step, frequency=0.02)
h_reference = signalGenerator(amplitude=h_step, frequency=0.03)

z_ref_hist = 5.0 + z_reference.square_batch(t_span)
h_ref_hist = 5.0 + h_reference.square_batch(t_span)
ref_hist = np.vstack( (z_ref_hist, h_ref_hist) )


# ======================================
# ======================================
# ======================================

nom_bount_tr = 3.0
pm_bound_tr = 1.0

nom_bount_zeta = 0.707
pm_bound_zeta = 0.1

lb_tr = nom_bount_tr - pm_bound_tr
up_tr = nom_bount_tr + pm_bound_tr

lb_tr_in = (nom_bount_tr - pm_bound_tr) * 0.1
up_tr_in = (nom_bount_tr + pm_bound_tr) * 0.1

lb_tr_ratio = 8
ub_tr_ratio = 100

lb_zeta = nom_bount_zeta - pm_bound_zeta
ub_zeta = nom_bount_zeta + pm_bound_zeta

lb_motor = 1.0
ub_motor = 15.0


# gains for starting point for optimization
kp_z = -4.e-02
ki_z = 0
kd_z = -2.e-02
kp_h = 1.
ki_h = .5
kd_h = 1.
kp_th = 4
kd_th = .5


# bounding values
wn_z_up     = 2.2/lb_tr
wn_z_low     = 2.2/up_tr

kp_z_up   = wn_z_up**2.0/b1
kp_z_low   = wn_z_low**2.0/b1
kd_z_up   = (2.0*lb_zeta*wn_z_up-a1)/b1
kd_z_low   = (2.0*ub_zeta*wn_z_low-a1)/b1




def rt_lon(pids):
    kp_h = pids[3]
    tr_h = 2.2 * np.sqrt( (mc+2*mr) / kp_h )
    return tr_h
    #
#
def rt_lat(pids):
    kp_z = pids[0]
    tr_z = 2.2 / (-gravity * kp_z)**0.5
    return tr_z
    #
#
def rt_th(pids):
    kp_th = pids[6]
    tr_th = 2.2 * np.sqrt( (Jc+2.0*mr*arm**2) / kp_th )
    return tr_th
    #
#
def rt_ratio(pids):
    return rt_lat(pids)/rt_th(pids)
    #
#
def zeta_lon(pids):
    kd_h = pids[5]
    zeta_h = (kd_h / ( 4.4 * (mc + 2*mr))) * rt_lon(pids)
    return zeta_h
    #
#
def zeta_lat(pids):
    kd_z = pids[2]
    zeta_z = (( -gravity * kd_z + ((mu)/(mc + 2*mr)) )
                    / 4.4) * rt_lat(pids)
    return zeta_z
    #
#
def zeta_th(pids):
    kd_th = pids[7]
    zeta_th = ( (kd_th / ( 4.4 * (Jc + 2 * mr * arm**2)))
                * rt_th(pids) )
    return zeta_th
    #
#
