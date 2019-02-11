# VTOL Parameter File
import numpy as np
# import control as cnt

# Physical parameters of the  VTOL known to the controller
mc = 1.0  # kg
mr = 0.25  # kg
Jc = 0.0042  # kg m^2
d = 0.3  # m
mu = 0.1  # kg/s
g = 9.81  # m/s^2
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
t_end = 150.0  # End time of simulation
Ts = 0.01  # sample time for simulation
t_plot = 0.1  # the plotting and animation is updated at this rate

# saturation limits
fmax = 10.0  # Max Force, N

# dirty derivative parameters
sigma = 0.05  # cutoff freq for dirty derivative
beta = (2.0*sigma-Ts)/(2.0*sigma+Ts)  # dirty derivative gain

# equilibrium force
Fe = (mc+2.0*mr)*g

# mixing matrix
mixing = np.linalg.inv(np.array([[1.0, 1.0],
                                  [d, -d]]))
