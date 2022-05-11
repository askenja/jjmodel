"""
Created on Tue Feb 23 15:44:21 2016

Physical constants in SI and CGS systems, as well as in astronomical units for convenient use. 

@author: Skevja
"""

import numpy as np 

# ==================================================================================================
# Physical constants and units
# ==================================================================================================

PC = 3.08e16                # m
M_SUN = 2e30                # kg
KM = 1e3                    # m
HR = 3600                   # s
YR = HR*24*365            # s
GYR = YR*1e9                # s 
G = 6.67e-11                # m^3/kg/s^2
GA = 6.67e-11*M_SUN/PC**3   # pc^3/M_SUN/GYR^2

# ==================================================================================================
# Model constants
# ==================================================================================================

# -------------------------------------------------------------------------------------------------
# The following quantites are used only as normalizations:
# SIGMA_E[kpc] - an estimate of the W-dispersion of the oldest thin-disk population. 
# RHOD0[Msun/pc^2] - an estimate of the local thin-disk density. 
# HEFFD[pc] - an estimate of the thin disk half-thickness. 
# ZE[pc] and ZN[pc] are 'natural scaleheights'
# -------------------------------------------------------------------------------------------------

SIGMA_E = 25    
RHO_D0 = 0.037  
HEFF_D = 400     
ZE = (SIGMA_E*KM)**2/(4*np.pi*G*(RHO_D0*M_SUN/PC**3)*(HEFF_D*PC))/PC     
ZN = np.sqrt(HEFF_D*ZE/2) 

# -------------------------------------------------------------------------------------------------
# tr[Gyr] and tp[Gyr] are the model time-resolution and the present-day MW disk age. 
# Theoretically, they can be changed, but it's not recommended, as the code may crash 
# (hopefully not, test this if you want). 
# -------------------------------------------------------------------------------------------------

tr = 0.025      
tp = 13   

# Sgr A* proper motion from Reid and Brunthaler (2005)
pm_SgrA = 6.37    # +-0.02 mas/yr 










