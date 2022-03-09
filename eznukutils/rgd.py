"""
helper functions related to rarefied gas dynamics
"""

import numpy as np
from . import physics as ph

def get_pin_pout(Kn_m, D_C, T, gas, d_gas=None, ratio=1000):
    """
    Calculates the inlet and outlet pressure using the Kn number.

    Parameters
    ----------
    Kn_m : float, ndarray
        Mean Knudsen numbers.
    D_C : float
        Characteristic length used calculating the Kn number.
    T : float, ndarray
        Temperature.
    gas : str
        Gas species.
    d_gas : float, optional
        Molecular diameter of the gas. If this is set, the diameter
        is used to calculate the pressures from the mean free path.
        Otherwise, the viscosity is used.
        The default is None.
    ratio : float, optional
        Ratio between inlet and outlet pressure. The default is 1000.

    Returns
    -------
    pin : float, ndarray
        Inlet pressures.
    pout : float, ndarray
        Outlet pressures.

    """
    mfp_m = Kn_m * D_C
    if d_gas:
        p_m = ph.kB*T/(np.sqrt(2)*np.pi*d_gas**2*mfp_m)
    else:
        p_m = mfp_to_p_visc(mfp_m, T, gas)
    pout = 2*p_m / (ratio+1)
    pin = pout * ratio
    return pin, pout

def mfp_to_p_visc(mfp, T, gas):
    """
    returns the pressure in Pa
    """
    p = ph.get_mu(gas, T) / mfp * np.sqrt(np.pi * ph.R * T 
                                       / (2*ph.get_M(gas)))
    return p

def mfp_visc(T, p, gas, M = None, mu = None):
    """
    Mean free path using kinematic viscosity. 
    Source: Sharipov, F. (1999). Rarefied gas flow through 
    a long rectangular channel 
    Journal of Vacuum Science & Technology A: Vacuum, 
    Surfaces, and Films  17(5), 3062-3066.
    https://dx.doi.org/10.1116/1.582006 [eq. (4)]
    (just using R and M instead of kB and m)
    
    Parameters
    ----------
    T : float
        temperature in K
    p : float
        pressure of gas in Pa
    gas : string
        gas name. If you want to specify M and mu manually, 
        set this to None.
    M : float, optional   
        molar mass in kg/mol
    mu : float, optional
        dynamic viscosity of gas in Pa*s
        
    Returns
    -------
    mfp : float
        mean free path in m
        
    """
    
    if gas:
        M = ph.get_M(gas)
        mu = ph.get_mu(gas, T)
    return mu/p * np.sqrt(np.pi * ph.R * T / (2*M))

def mfp(T,d,p):
    """
    mean free path using particle diameter
    source: Brodkey, Hershey: Transport Phenomena, Volume 2, p. 716
    
    Parameters
    ----------
    T : float, ndarray
        temperature in K
    d : float
        gas molecule diameter in m
    p : float, ndarray
        gas pressure in Pa
        
    Returns
    -------
    mfp : float, ndarray
        mean free path in m
    """        
    return ph.kB*T/(np.sqrt(2)*np.pi*d**2*p)


def get_Kn(pin, pout, T, d_h, gas, d_gas=None):
    if d_gas:
        mfp_m = mfp(T, d_gas, (pin+pout)/2)
    else:
        mfp_m = mfp_visc(T, (pin+pout)/2, gas)
    Kn_m = mfp_m / d_h
    return Kn_m

def mdot_to_g(mdot, L, P, A, dp, T, gas):
    """
    Calculates the dimensionless mass flow.    

    Parameters
    ----------
    mdot : float, ndarray
        Mass flow.
    L : float
        Length of the channel.
    P : float
        Perimeter of the channel.
    A : float
        Cross-sectional area of the channel.
    dp : float, ndarray
        Pressure difference between inlet and outlet.
    T : float, ndarray
        Temperature.
    gas : str
        Gas species.

    Returns
    -------
    G : float, ndarray
        Dimensionless mass flow.

    """
    M = ph.get_M(gas)
    G = mdot * 3*P*L / (8*A**2*dp) * np.sqrt(np.pi*ph.R*T/(2*M))
    return G


# ----------------------------------------------
# --- more precise sources for the following ---
# ----------------------------------------------

def efp(T,M,mu,p):
    """Equivalent free path according to eqs (1.32) and (2.40) in:
    Rarefied Gas Dyanmics, Sharipov
    
    Parameters
    ----------
    T : float
        temperature in K
    M : float    
        molar mass in kg/mol
    mu : float
        dynamic viscosity of gas in Pa*s
    p : float
        pressure of gas in Pa
        
    Returns
    -------
    mfp : float
        mean free path in m
    """
    return mu / p * np.sqrt(2*ph.R*T/M)

def mvel(T,M):
    """most probable molecular velocity according to the
    Maxwellian distribution.
    
    Parameters
    ----------
    T : float, ndarray
        temperature in K
    M : float, ndarray
        molar mass in kg/mol
        
    Returns
    -------
    mvel : float, ndarray
        most probable molecular velocity in m/s
    """
    return np.sqrt(2*ph.R*T/M) 


def mdot_to_g_veltzke(mdot, T, l, h, w, pin, pout, M):
    """
    calculate dimensionless mass flow according to 
    Veltzke's Diss, p. 45,  eq. (3.80) for rectangular channels
    
    Parameters
    ----------
    mdot : float
        mass flow in kg/s
    T : float
        temperature in K
    l, h, w : float
        dimensions of channel in m
    pin, pout : float
        inlet & outlet pressures in Pa
    M : float
        molar mass in kg/mol
        
    Returns
    -------
    G : float
        dimensionless mass flow
    """
    P = 2*w + 2*h
    A = w*h
    G = mdot * 3*P*l / (4*A**2*(pin-pout)) * np.sqrt(ph.R*T/(2*M))
    
    return G

    """
    calculate dimensionless mass flow according to eq. (10) in
    Graur et al: Measurements of tangential momentum, 2009,
    Phys. Fluids
    R = 8.3144598   #universal gas constant [kg m^2 / (s^2 mol K)]
    G = l * np.sqrt(2*R*T) / (h**2 * w * (pin-pout)) * mdot / M
    """
    
def g_rect_to_mdot(G, T, l, h, w, pin, pout, M):
    """
    inverse function of mdot_to_g() for rectangular channels
    """
    P = 2*w + 2*h
    A = w*h
    mdot = G / (3*P*l / (4*A**2*(pin-pout)) * np.sqrt(ph.R*T/(2*M)))
    return mdot
    

def mdot_to_g_pipe(mdot, T, D, l, pin, pout, M):
    """
    calculate dimensionless mass flow by
    normalizing to Kn Diff mass flow (eq. 3.49, p. 35
    in Veltzke Diss)
    """
    A = np.pi/4 * D**2
    P = np.pi * D
    dp = pin-pout
    mdot_Kn = 4*A**2*dp/(3*P*l) * np.sqrt(2*M/(ph.R*T))
    return mdot / mdot_Kn

def g_pipe_to_mdot(g, T, D, l, pin, pout, M):
    """
    reverseve function of mdot_to_g_pipe()
    """
    A = np.pi/4 * D**2
    P = np.pi * D
    dp = pin-pout
    mdot_Kn = 4*A**2*dp/(3*P*l) * np.sqrt(2*M/(ph.R*T))
    return g * mdot_Kn

def mdot_to_q_bar_karniadakis(mdot, T, l, h, w, pin, pout, M):
    """
    calculate the dimensionless flow according to
    Karniadakis & Beskok: Microflows and Nanoflows: Fundamentals and Simulation,
    p. 144
    """
    R_s = ph.R/M
    p_m = 0.5*(pin + pout)
    Q_dot = (mdot * R_s*T / p_m) / w
    Q_bar = Q_dot * p_m / ((pin-pout)/l * h**2 * (R_s*T)**0.5)
    return Q_bar

def mdot_to_g_veltzke_general(mdot, L, P, A, dp, T, gas):
    """
    Veltzke's Diss, eq. (3.80)
    P: perimeter
    A: cross-sectional area
    """
    M = ph.get_M(gas)
    G = mdot * 3*P*L / (4*A**2*dp) * np.sqrt(ph.R*T/(2*M))
    return G

def g_to_mdot_veltzke_general(G, L, P, A, dp, T, gas):
    """
    Veltzke's Diss, eq. (3.80)
    P: perimeter
    A: cross-sectional area
    """
    M = ph.get_M(gas)
    mdot = G * 4*A**2*dp / (3*P*L) * np.sqrt(2*M/(ph.R*T))
    return mdot
    
    
def G_orifice_to_mdot(G, h, w, l, gas, T, dp):
    """calculate mdot from G. The G is the dimensionless mass flow
    relative to the analyitical solution of the mass flow through
    an orifice (according to Stelios' excel sheet)
    This will result in the mass flow for just one channel!
    """
    D_h = 2*h*w / (w+h)     # hydraulic diameter
    A = h*w                 # diameter
    R_s = ph.R / ph.get_M(gas)
    u0 = np.sqrt(2*R_s*T)
    mdot = G * D_h * A * dp / (l * u0)
    return mdot

def D_H(w, h):
    """ returns the hydraulic diameter of a rectangular duct"""
    return 2*h*w / (h+w)

def get_pin_pout(Kn_m, D_H, T, gas, d_gas=None, ratio=1000):
    mfp_m = Kn_m * D_H
    if d_gas:
        p_m = ph.kB*T/(np.sqrt(2)*np.pi*d_gas**2*mfp_m)
    else:
        p_m = mfp_to_p_visc(mfp_m, T, gas)    
    pout = 2*p_m / (ratio+1)
    pin = pout * ratio
    return pin, pout

def get_transition_diamater(gas, ret_var=False):
    if gas == "He":
        d = 209e-12
        var = 3e-12
    elif gas == "N2":
        d = 369e-12
        var = 9e-12
    elif gas == "Ar":
        d = 317e-12
        var = 3e-12
    elif gas == "CO2":
        d = 419e-12
        var = 8e-12
    else:
        raise ValueError(f"no valid gas given: {gas=}")
    
    if ret_var:
        return d, var
    else:
        return d
