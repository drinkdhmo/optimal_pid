from IPython.core.debugger import set_trace

import autograd.numpy as np

from importlib import reload

import VTOLParam as Param
import VTOLSim as sim

reload(Param)
reload(sim)

# ======================================
# ======================================

# constraint eqs for height

tr_h = 2.2 * np.sqrt( (mc+2*mr) / kp_h )

zeta_h / tr_h = kd_h / ( 4.4 * (mc + 2*mr))

2.7 < tr_h < 3.3

0.65 < zeta_h < 0.716

# ======================================

# constraint eqs for theta

tr_th = 2.2 * np.sqrt( (Jc+2.0*mr*arm**2) / kp_th )

zeta_th = (kd_th / ( 4.4 * (Jc + 2 * mr * arm**2))) * tr_th

# needs to be 10x faster than z (lateral) loop
0.27 < tr_th < 0.33

0.65 < zeta_th < 0.716

# ======================================

# constraint eqs for z (lateral)

tr_z = -(2.2**2) / (gravity * kp_z**2)

zeta_z = (( -gravity * kd_z + ((mu)/(mc + 2*mr)) ) / 4.4) * tr_z

2.7 < tr_z < 3.3

0.65 < zeta_z < 0.716


# ======================================

# constrain each motor thrust, theta,

#
