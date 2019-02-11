import numpy as np
import random
import VTOLParam as P


class VTOLDynamics:
    '''
        Model the physical system
    '''

    def __init__(self):
        # Initial state conditions
        self.state = np.array([[P.z0],          # initial lateral position
                                [P.h0],          # initial altitude
                                [P.theta0],      # initial roll angle
                                [P.zdot0],       # initial lateral velocity
                                [P.hdot0],       # initial climb rate
                                [P.thetadot0]])  # initial angular velocity
        #################################################
        # The parameters for any physical system are never known exactly.  Feedback
        # systems need to be designed to be robust to this uncertainty.  In the simulation
        # we model uncertainty by changing the physical parameters by a uniform random variable
        # that represents alpha*100 % of the parameter, i.e., alpha = 0.2, means that the parameter
        # may change by up to 20%.  A different parameter value is chosen every time the simulation
        # is run.
        alpha = 0.2  # Uncertainty parameter
        self.mc = P.mc * (1+2*alpha*np.random.rand()-alpha)
        self.mr = P.mr * (1+2*alpha*np.random.rand()-alpha)
        self.Jc = P.Jc * (1+2*alpha*np.random.rand()-alpha)
        self.d = P.d * (1+2*alpha*np.random.rand()-alpha)
        self.mu = P.mu * (1+2*alpha*np.random.rand()-alpha)
        self.F_wind = P.F_wind * (1+2*alpha*np.random.rand()-alpha)

    def propagateDynamics(self, u):
        '''
            Integrate the differential equations defining dynamics
            P.Ts is the time step between function calls.
            u contains the system input(s).
        '''
        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = self.derivatives(self.state, u)
        k2 = self.derivatives(self.state + P.Ts/2*k1, u)
        k3 = self.derivatives(self.state + P.Ts/2*k2, u)
        k4 = self.derivatives(self.state + P.Ts*k3, u)
        self.state += P.Ts/6 * (k1 + 2*k2 + 2*k3 + k4)

    def derivatives(self, state, u):
        '''
            Return xdot = f(x,u), the derivatives of the continuous states, as a matrix
        '''
        # re-label states and inputs for readability
        z = state.item(0)
        h = state.item(1)
        theta = state.item(2)
        zdot = state.item(3)
        hdot = state.item(4)
        thetadot = state.item(5)
        fr = u[0]
        fl = u[1]
        # The equations of motion.
        zddot = (-(fr + fl) * np.sin(theta) + self.F_wind) / (self.mc + 2.0*self.mr)
        hddot = (-(self.mc + 2.0*self.mr) * P.g + (fr + fl) * np.cos(theta)) / (self.mc + 2.0*self.mr)
        thetaddot = self.d * (fr - fl) / (self.Jc + 2.0*self.mr*(self.d**2))
        # build xdot and return
        xdot = np.array([[zdot], [hdot], [thetadot], [zddot], [hddot], [thetaddot]])
        return xdot

    def outputs(self):
        '''
            Returns the measured outputs as a list
            [z, h, theta] with added Gaussian noise
        '''
        # re-label states for readability
        z = self.state.item(0)
        h = self.state.item(1)
        theta = self.state.item(2)
        # add Gaussian noise to outputs
        z_m = z + random.gauss(0, 0.001)
        h_m = h + random.gauss(0, 0.001)
        theta_m = theta + random.gauss(0, 0.001)
        # return measured outputs
        return [z_m, h_m, theta_m]

    def states(self):
        '''
            Returns all current states as a list
        '''
        return self.state.T.tolist()[0]
