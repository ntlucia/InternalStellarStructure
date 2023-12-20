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


##################################################################

## Functions to implement the solution of the stellar structure ##
## of a star with masses > 1.3 Solar Masses, also assuming      ##
## solar composition X=0.7, Y=0.28, Z=0.02                      ##

##################################################################


#Definition of density
def density(P,T,X=0.7):
    mu = 4/(3+5*X)
    return (P - ((c.a/3)*T**4))*mu/(N_A * c.k*T)


#read opacities
rosseland = pd.read_csv('./opacities/combined_OPAL_F05_X0.7_Z0.02_logT8.7-2.7.txt', index_col=0)
#Here we assume the solar composition X=0.7, Y=0.28, Z=0.02
rosseland.columns = rosseland.columns.astype(float)


# setup the grid
points = []
values = []
for i in range(len(rosseland.columns.values)):
    for j in range(len(rosseland.index.values)):
        points.append([rosseland.columns.values[i], rosseland.index.values[j]])
        values.append(rosseland.values[j][i])


def interp_k(rho_i, T_i, method='linear'):
    '''
    Function to interpolate across the Rosseland
    opacity grid using scipy.interpolate.interpn
    '''
    # calc logR (R=rho/T_6^3)
    logR_i = np.log10(rho_i/(T_i/1e6)**3)
    # calc logT
    logT_i = np.log10(T_i)

    # arrange interpolation point in array
    new_point = (logT_i,logR_i)
    # compute interpolation
    interpolated_points = 10**griddata(points, values, (logR_i, logT_i), method=method)
    return interpolated_points


del_ad = 0.4 # assuming ideal gas, fully ionization

def del_rad(m, l, P, rho, T):
    """
    Funtion to calculate nabla_rad based on 4.30 in HKT
    """
    kappa = interp_k(rho, T, method='linear')
    return (3/(16*np.pi*c.a*c.c))*(P*kappa/T**4)*(l/(c.G*m))



#Determinate the energy generate in core, where total energy is the sum of CNO-cycle/pp-chain contributions

##we based these calculation in the equations of the book: Kippenhahn, R https://doi.org/10.1088/0067- 0049/192/1/3 (KWW)

def poor_mans_psi(T):
    '''
    A piecewise approximation of the pp-chain He dependency, fit by eye from
    figure 18.7 in KWW
    '''
    T7 = T / 1e7
    if T7 < 1.:
        p = 1.
    elif T7 > 3.:
        p = 1.4
    else:
        p = .2 * T7 + 0.8
    return p

#pp chain
def epsilon_pp(rho,T, X):
    '''
    Energy generation for the pp-chain, from KWW Ch 18.4
    '''
    T9 = T/1e9 # assuming at the beginning based on our opacities

    #f parameter
    T7 = T/1e7
    term1 = 5.92e-3
    Z1Z2 = 1 # assuming pp-chain
    zeta = 1 # KWW say this is of order unity
    term2 = (zeta*rho/(T7**3))**(1/2)
    edkt = term1*Z1Z2*term2
    f_weak =  np.exp(edkt)

    try:
        psi_T = poor_mans_psi(T)
    except ValueError:
        psi_T = np.zeros_like(T)
        for i,t in enumerate(T):
            psi_T[i] += poor_mans_psi(t)
    
    g_pp = 1 + (3.82* T9) + (1.51* T9**2) + (0.144* T9**3) - (0.0114* T9**4)
    
    return 2.57e4 * psi_T * f_weak * g_pp * rho * X**2 * T9**(-2/3) * np.exp(-3.381/(T9**(1/3)))


#CNO cycle
def epsilon_cno(rho,T,Z, X):
    '''
    Energy generation rate for the CNO cycle, from KWW Ch 18.4
    '''
    T9 = T/1e9
    X_cno = (2/3)*Z
    g_cno = 1 - (2.00*T9) + (3.41*T9**2) - (2.43*T9**3)
    return 8.24e25*g_cno*X_cno*X*rho*T9**(-2/3)*np.exp((-15.231*T9**(-1/3))-(T9/0.8)**2)

#Total energy
def energy_gen(rho,T,X=0.7,Z=0.02):
    '''
    Calculate rate of energy generation in cgs following KWW Ch 18.5, which draws from Angulo+99
    '''
    e_pp = epsilon_pp(rho,T,X)
    e_cno = epsilon_cno(rho,T,Z,X)
    return e_pp+e_cno

# Definition of the ordinary differential equations, based in the four structure equations: hydrostatic eqillibrium, mass conservation, energy generation, and transport energy

def ode(m,v):
    """
    The four coupled differential equations for a simple stellar model
    See KWW Ch 10 or HKT Ch 7.
    """

    # load vector of variables
    l, P, r, T = v
    # calculate density
    rho = density(P,T)
    # calculate del_rad
    del_radiative = del_rad(m, l, P, rho, T)
    del_ad = 0.4
    # determine whether the star is convective or not
    del_actual = np.minimum(del_radiative, del_ad)

    # calculate the derivatives
    dldm = energy_gen(rho,T,0.7,0.02) #change in luminosity with enclosed mass
    dPdm = -c.G*m/(4*np.pi*r**4) # change in pressure with enclosed mass
    drdm = 1/(4*np.pi*r**2 * rho) # mass conservation eq.
    dTdm = ((-c.G*m*T)/(4*np.pi*P*r**4))*del_actual # change in temperature with enclosed mass

    # return derivatives
    return np.array([dldm, dPdm, drdm, dTdm])


## Define the grid of the internal and sourface part, take into account the boundary conditions

def load_inner(Tc, Pc, factor=1e-12):
    """
    Returns reasonable guess for an integration
    starting point exterior to the absolute center
    """

    rho_c = density(Pc,Tc, X=0.7) # calculate the density of the core

    m = factor*c.Ms # set start just outside center in M_r
    epsilon = energy_gen(rho_c,Tc) # determine energy generation from pp and CNO
    l = epsilon*m # calculate luminosity at core

    # KWW eq. 11.3
    r = (3/(4*np.pi*rho_c))**(1/3) * m**(1/3) # radius starting point

    # KWW eq. 11.6
    P = Pc - (3*c.G/(8*np.pi))*((4*np.pi*rho_c/3)**(4/3))*(m**(2/3)) # pressure just outside core

    # calculate temperature gradient
    delrad = del_rad(m, l, P, rho_c, Tc)

    # determine whether the core is convective or radiative
    if delrad > del_ad:
        # if convective, use KWW eq. 11.9 b
        # see also HKT eq. 7.110
        lnT = np.log(Tc) - (np.pi/6)**(1/3)*c.G*(del_ad*rho_c**(4/3))/Pc*m**(2/3)
        T = np.exp(lnT)
    else:
        # if radiative, use KWW eq. 11.9 a
        # see also HKT eq. 7.109
        kappa_c = interp_k(rho_c,Tc)
        T = (Tc**4 - (1/(2*c.a*c.c)*(3/(4*np.pi))**(2/3)*kappa_c*epsilon*rho_c**(4/3)*m**(2/3)))**(1/4)

    # return guess array
    return np.array([l, P, r, T])


def load_outer(M_star, L_star, R_star, factor=0.9999, X=0.7):
    """
    Returns reasonable guess for an integration
    starting point interior to the photosphere.
    """

    # set mu based on hydrogen fraction
    mu = 4/(3+5*X)
    # calculate surface gravity from mass, radius
    surface_g = c.G*M_star/(R_star**2)
    # calculate Teff from luminosity, radius
    Teff = (L_star/(4*np.pi*c.sb*R_star**2))**(1/4)

    # minimize the difference between rho based on opacity and rho based on equation of state
    # we need to find this rho so we can determine the photospheric opacity
    # and therefore the surface pressure, which is our final guess component
    def min_rho(rho):
        kappa = interp_k(rho,Teff)
        # eq. 4.48 in HKT
        opacity_pressure = (2/3) * (surface_g / kappa) #* (1 + (kappa*L_star/(4*np.pi*c.c*c.G*M_star)))
        gas_rad_pressure = (1/3)*c.a*Teff**4 + (rho * N_A*c.k*Teff/mu)
        diff = 1 - opacity_pressure/gas_rad_pressure
        return np.abs(diff**2)
    # minimize this difference
    rho_sol = optimize.minimize(min_rho, 1e-8, args=(), method='Nelder-Mead', bounds=[(1e-13,1e-5)])
    # determine whether the minimizer was successful
    if rho_sol.success:
        rho = rho_sol.x[0]
    else:
        # return a nan if we interpolate off the grid or have some weird negative value
        print('there\'s no rho for this Teff, log(g)')
        rho = np.nan
    # calculate surface opacity and pressure
    kappa = interp_k(rho,Teff)
    P = 2*surface_g/(3*kappa) #* (1 + (kappa*L_star/(4*np.pi*c.c*c.G*M_star)))
    # return guess array
    return np.array([L_star, P, R_star, Teff])


def shooter(vec, M_star=1.67*c.Ms, M_fit=0.5,
            n=int(1e5), in_factor=1e-12, out_factor=0.9999,
            multithread=False, number_nuc=4
            ):
    """
    This is a version of the "shootf" function that takes a vector of initial guesses
    (luminosity, central pressure, radius, central temperature) and shoots towards
    a solution from the interior and surface.
    set mass of star with M_star
    set start points (in fraction of enclosed mass) via in_factor and out_factor
    set fitting point (in fraction of enclosed mass) via M_fit
    Multithread the ODE integration using ray/pool can be toggled (for more effiency)
    """

    # load in vector of initial guess variables
    L_star, Pc, R_star, Tc = vec

    # load initial guess vectors based on input variables
    inn = load_inner(Tc, Pc, factor=in_factor)
    outt = load_outer(M_star, L_star, R_star, factor=out_factor)

    # protect against bad solutions which crash the minimizer
    if np.isnan(np.sum(inn)) or np.isnan(np.sum(outt)):
        print('caught a nan in the guess')
        return np.array([-np.inf, -np.inf, -np.inf, -np.inf])

    # set up array of enclosed mass to solve across
    exiting = np.logspace(np.log10(in_factor*c.Ms), np.log10(M_fit*M_star), base = 10.0, num = int(n))
    entering = np.append(np.flipud(np.linspace(M_star*out_factor, M_star, num = int(n/2))), np.flipud(np.linspace(M_fit*M_star, M_star*out_factor, num = int(n/2)))[1:])

    # set up multithreading
    if multithread:
        ray.init(num_cpus=number_nuc)
        pool = Pool()
        # solve heading from core to surface
        sol_i = pool.apply(solve_ivp, [ode, (exiting[0], exiting[-1]), inn, 'RK45', exiting])
    else:
        # solve heading from core to surface
        sol_i = solve_ivp(ode, (exiting[0], exiting[-1]), inn, method='RK45', t_eval=exiting)
    # determine success of core->surface integrator
    if sol_i.status == 0:
        # report success
        print('solved inner')
    else:
        # report failure, shutdown multithread if using multithreading
        print('failed to solve interior', sol_i.message)
        if multithread:
            ray.shutdown()
        # protect minimizer against failed solutions
        return np.array([-np.inf, -np.inf, -np.inf, -np.inf])

    if multithread:
        # solve heading from surface to core
        sol_s = pool.apply(solve_ivp, [ode, (entering[0], entering[-1]), outt, 'RK45', entering])
    else:
        # solve heading from surface to core
        sol_s = solve_ivp(ode, (entering[0], entering[-1]), outt, method='RK45', t_eval=entering)
    # determine success of core->surface integrator
    if sol_s.status == 0:
        # report success
        if multithread:
            ray.shutdown()
        print('solved exterior')
    else:
        # report failure, shutdown multithread if using multithreading
        print('failed to solve exterior', sol_s.message)
        if multithread:
            ray.shutdown()
        # protect minimizer against failed solutions
        return np.array([-np.inf, -np.inf, -np.inf, -np.inf])

    # assign integrated solution to variables
    exiting_sol = sol_i.y
    entering_sol = sol_s.y

    # determine the difference at the shooting point
    dL = (exiting_sol[0,-1] - entering_sol[0,-1])/L_star
    dP = (exiting_sol[1,-1] - entering_sol[1,-1])/Pc
    dR = (exiting_sol[2,-1] - entering_sol[2,-1])/R_star
    dT = (exiting_sol[3,-1] - entering_sol[3,-1])/Tc
    # return residual array
    print(np.array([dL, dP, dR, dT]))
    return np.array([dL, dP, dR, dT])


def solver(vec_final, M_star=1.67*c.Ms, M_fit=0.5,
            n=int(1e5), in_factor=1e-12, out_factor=0.9999,
            multithread=True,
            ):
    """
    a version of the "shootf" function that solves the ode given the results of a
    minimizer and returns a final solution array
    set mass of star with M_star
    set start points (in fraction of enclosed mass) via in_factor and out_factor
    set fitting point (in fraction of enclosed mass) via M_fit
    Multithread the ODE integration using ray/pool can be toggled
    """

    L_star, Pc, R_star, Tc = vec_final

    inn = load_inner(Tc, Pc, factor=in_factor)
    outt = load_outer(M_star, L_star, R_star, factor=out_factor)

    exiting = np.logspace(np.log10(in_factor*c.Ms), np.log10(M_fit*M_star), base = 10.0, num = int(n))
    entering = np.append(np.flipud(np.linspace(M_star*out_factor, M_star, num = int(n/2))), np.flipud(np.linspace(M_fit*M_star, M_star*out_factor, num = int(n/2)))[1:])

    ray.init(num_cpus=4)
    pool = Pool()
    sol_i = pool.apply(solve_ivp, [ode, (exiting[0], exiting[-1]), inn, 'RK45', exiting])
    sol_s = pool.apply(solve_ivp, [ode, (entering[0], entering[-1]), outt, 'RK45', entering])
    ray.shutdown()
    exiting_sol = sol_i.y
    entering_sol = sol_s.y

#  machine-readable table with Lagrangian mass coordinate m, radius r, den-
# sity ρ, temperature T , pressure P , luminosity l, nuclear energy generation
# rate , opacity κ, adiabatic temperature gradient ∇ad, actual temperature
# gradient ∇ = d ln T /d ln P , and the convective/radiative nature of the shell;

    # combine mass arrays
    mass = np.concatenate([exiting, np.flipud(entering)], axis=0)

    # add mass to final array
    solution = np.zeros((10, mass.shape[0]))
    solution[0] = mass

    # combine solution arrays
    sols = np.concatenate([exiting_sol, np.fliplr(entering_sol)], axis=1)
    solution[1:5] = sols

    # add density as 6th column
    rho = density(solution[2],solution[4], X=0.7)
    solution[5] = rho

    # add energy generation rate
    epsilon = energy_gen(rho,solution[4])
    solution[6] = epsilon

    kappa = interp_k(rho,solution[4])
    solution[7] = kappa

    del_ad = 0.4
    # add del ad
    delad = np.zeros_like(epsilon) + del_ad
    solution[8] = delad

    # add del_rad
    delrad = del_rad(mass, solution[1], solution[2], rho, solution[4])
    solution[9] = delrad

    return solution

