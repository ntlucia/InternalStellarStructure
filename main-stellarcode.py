import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from scipy import optimize

import ray
from ray.util.multiprocessing import Pool

# constants
import constants as c
from scipy.constants import N_A

#funtions
import functions as f
from functions import density
from functions import interp_k as interpolate
from functions import energy_gen as energy
from functions import load_inner, load_outer 
from functions import ode
from functions import shooter, solver

##################################################################
##                                                              ##
## Execute code to implement the solution of the stellar        ##
## structure of a star with masses > 1.3 Solar Masses,          ##
## also assuming solar composition X=0.7, Y=0.28, Z=0.02        ##
##                                                              ##
##################################################################


############# Here you can modify ##############

#Initial parameters
M_star = 1.33*c.Ms #Stellar mass

#Shootf args: Star mass, Shooting point in fraction of Mass, number of saved points, interior starting point, exterior starting point, do multiprocessing? number of cpus to use
args = (M_star, 0.25, int(1e5), 1e-12, 0.9999, True, 4)


########### Here you can NOT modify ############

X = 0.7 #Fraction of Hydrogen
Y= 0.28 #Fraction of Helium
Z=0.02 #Fraction of the other elements
mu = 4./(3+5*X) #mean molecular weight assuming fully ionized



#Surface
R_star_starting = ((M_star/c.Ms)**(0.75))*c.Rs # eq. 1.87 from Stellar Interiors - C. Heansen
L_star_starting = ((M_star/c.Ms)**(3.5))*c.Ls # eq. 1.88 from Stellar Interiors - C. Heansen


#Core
Pc_starting = (3/(8*np.pi))*(c.G*(M_star)**2)/(R_star_starting)**4 # Lower limit, using the constant density model (eq. 1.42)
Tc_starting = (((1/2)*mu)/(N_A*c.k))*(c.G*M_star)/(R_star_starting) # From equilibrium equation and assuming ideal gas

print('Starting Guess:')
print('L/Lsun', L_star_starting/c.Ls)
print('R/Rsun', R_star_starting/c.Rs)
print('logPc', np.log10(Pc_starting))
print('logTc', np.log10(Tc_starting))

# initial guess vector
vec = np.array([L_star_starting, Pc_starting, R_star_starting, Tc_starting])

# set limits for the minimizer
bounds = (np.array([1e-1*c.Ls, Pc_starting, 1e-1*c.Rs, Tc_starting]),
          np.array([1e6*c.Ls, Pc_starting*1e3, 1e3*c.Rs, Tc_starting*1e2]))

final = least_squares(shooter, vec, args=args, bounds=bounds,
                      method='dogbox', loss='arctan',
                      gtol=None,
                      xtol=None,
                      ftol=1e-6,
                      x_scale='jac',
                     )

print(final)

if np.sum(final.active_mask**2) != 0:
    print('something ran up against a bound')

# run solution and create densely sampled results table
solution = solver(final.x, M_star=args[0], M_fit=args[1], n=1e6, in_factor=args[3], out_factor=args[4], multithread=args[5])

#Results
L_star, Pc, R_star, Tc = final.x
print('Converged starting values:')
print('L/Lsun', L_star/c.Ls)
print('logPc', np.log10(Pc))
print('R/Rsun', R_star/c.Rs)
print('logTc', np.log10(Tc))

print('T_eff',((L_star/(4*np.pi*c.sb*R_star**2))**(1/4)))
print('logg', (np.log10(c.G*M_star/(R_star**2))))

#saved the parameters in a txt
np.savetxt('converged_start_{}.txt'.format(M_star/c.Ms), final.x)

# what is an appropriate P_factor to speed up convergence?
print('ratio between constant density Pc and converged solution',Pc/Pc_starting)

print('ratio between constant density Tc and converged solution',Tc/Tc_starting)

# central density vs avg density
converged_concentration = solution[5].max()/(4*np.pi*M_star/(3*R_star**3))

print("rho_c/rho_avg is", str(round(converged_concentration,2)))

# save dense results table to disk
with open('converged_interior_{}.npy'.format(M_star/c.Ms), 'wb') as f:
    np.save(f, solution)
