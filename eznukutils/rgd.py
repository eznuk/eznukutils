"""
Utilitiy functions related to rarefied gas dynamics.
All units in SI unless otherwise stated.
"""

import numpy as np
from typing import Tuple

from . import physics as ph
from .types import NumericArrayLike as NAL, StringIterable


def get_pin_pout(Kn_m: NAL, D_C: float, T: NAL, gas: str,
                 d_gas: float = 0, ratio: float = 1000) -> Tuple[NAL, NAL]:
    """
    Calculate the inlet and outlet pressure using the Kn number.

    Args:
        Kn_m: Mean Knudsen numbers.
        D_C: Characteristic length used calculating the Kn number.
        T: Temperature.
        gas: Gas species.
        d_gas: Molecular diameter of the gas. If this is set, the diameter
            is used to calculate the pressures from the mean free path.
            Otherwise, the viscosity is used. Defaults to 0.
        ratio: Ratio between inlet and outlet pressure. Defaults to 1000.

    Returns:
        pin: Inlet pressures.
        pout: Outlet pressures.

    """
    mfp_m = Kn_m * D_C
    if d_gas:
        p_m = ph.kB*T/(np.sqrt(2)*np.pi*d_gas**2*mfp_m)
    else:
        p_m = mfp_to_p_visc(mfp_m, T, gas)
    pout = 2*p_m / (ratio+1)
    pin = pout * ratio
    return pin, pout

def mfp_to_p_visc(mfp: NAL, T: NAL, gas: str):
    """Calculate the pressure using the mean free path. Use the formula
    used when calculating the mean free path from viscosity.
    
    Args:
        mfp: Mean free path.
        T: Temperature.
        gas: Gas species.
    
    Returns:
        The pressure in Pa
    """
    mfp_at_p_1 = mfp_visc(T, 1, gas)
    return mfp_at_p_1 / mfp

def mfp_visc(T: NAL, p: NAL, gas: StringIterable, 
             M: NAL = None, mu: NAL = None):
    """
    Mean free path using kinematic viscosity. 
    Source: Sharipov, F. (1999). Rarefied gas flow through 
    a long rectangular channel 
    Journal of Vacuum Science & Technology A: Vacuum, 
    Surfaces, and Films  17(5), 3062-3066.
    https://dx.doi.org/10.1116/1.582006 [eq. (4)]
    (just using R and M instead of kB and m)
    
    Args:
        T: Temperature.
        p: Pressure.
        gas: Gas species. If you want to specify M and mu manually, 
            set this to None.
        M: Molar mass in kg/mol. Defaults to None.
        mu: Dynamic viscosity of gas in Pa*s. Defaults to None.
        
    Returns:
        Mean free path in m.
    """
    
    if gas:
        M = ph.get_M(gas)
        mu = ph.get_mu(gas, T)
    return mu/p * np.sqrt(np.pi * ph.R * T / (2*M))

def mfp(T: NAL, d: float, p: NAL):
    """Calculate mean free path using molecular diameter.
    source: Brodkey, Hershey: Transport Phenomena, Volume 2, p. 716
    
    Args:
        T: Temperature.
        d: Gas molecule diameter.
        p: Gas pressure.
        
    Returns:
        Mean free path.
    """        
    return ph.kB*T/(np.sqrt(2)*np.pi*d**2*p)


def get_Kn(pin: NAL, pout: NAL, T: NAL, D_C: float, 
           gas: StringIterable, d_gas: float = None) -> NAL:
    """Calculate the Knudsen number given the inlet and outlet pressures.
    If a gas diameter is given, this diameter will be used to calculate
    the mean free path. Otherwise, the mean free path calculated
    from viscosity will be used.
    
    Args:
        pin: Inlet pressure.
        pout: Outlet pressure.
        T: Temperature.
        D_C: Characteristic length.
        gas: Gas species.
        d_gas: Gas diameter. Defaults to None.
        
    Returns:
        Knudsen number.
    """
    if d_gas:
        mfp_m = mfp(T, d_gas, (pin+pout)/2)
    else:
        mfp_m = mfp_visc(T, (pin+pout)/2, gas)
    Kn_m = mfp_m / D_C
    return Kn_m

def mdot_to_g(mdot: NAL, L: float, P: float, A: float,
              dp: NAL, T: NAL, gas: StringIterable) -> NAL:
    """Calculate the dimensionless mass flow.    

    Args:
        mdot: Mass flow.
        L: Length of the channel.
        P: Perimeter of the channel.
        A: Cross-sectional area of the channel.
        dp: Pressure difference between inlet and outlet.
        T: Temperature.
        gas: Gas species.

    Returns:
        Dimensionless mass flow.
    """
    M = ph.get_M(gas)
    G = mdot * 3*P*L / (8*A**2*dp) * np.sqrt(np.pi*ph.R*T/(2*M))
    return G

def mvel(T: NAL, M: NAL) -> NAL:
    """Most probable molecular velocity according to the
    Maxwellian distribution.
    
    Args:
        T: Temperature in K
        M: Molar mass in kg/mol
        
    Returns:
        Most probable molecular velocity in m/s.
    """
    return np.sqrt(2*ph.R*T/M) 

def mdot_to_g_veltzke(mdot: NAL, L: float, P: float, A: float, 
                     dp: NAL, T: NAL, gas: str) -> NAL:
    """Calculate dimensionless mass flow for rectangular channels
    according to 

    Veltzke, T. On Gaseous Microflows Under Isothermal Conditions. (2013).
    p. 45,  eq. (3.80)

    Args:
        mdot: Mass flow in kg/s.
        L: Length of channel.
        P: Perimeter of channel.
        A: Cross-sectional area of channel.
        dp: Pressure difference between inlet and outlet.
        T: Temperature.
        gas: Gas species.

    Returns:
        Dimensionless mass flow.        
    """
    M = ph.get_M(gas)
    G = mdot * 3*P*L / (4*A**2*dp) * np.sqrt(ph.R*T/(2*M))
    return G

def mdot_to_g_graur(mdot: NAL, T: NAL, l: float, h: float, w: float,
                      pin: NAL, pout: NAL, M: float) -> NAL:
    """Calculate dimensionless mass flow according to eq. (10) in
    
    Graur, I. A., Perrier, P., Ghozlani, W. & Méolans, J. G.
    Measurements of tangential momentum accommodation coefficient
    for various gases in plane microchannel. Phys Fluids 21, 102004 (2009).
  
    Args:
        mdot: Mass flow in kg/s.
        T: Temperature.
        l, h, w: Dimensions of channel.
        pin, pout: Inlet & outlet pressures.
        M: Molar mass in kg/mol.

    Returns:
        Dimensionless mass flow.
    """
    return l * np.sqrt(2*ph.R*T) / (h**2 * w * (pin-pout)) * mdot / M

def mdot_to_q_bar_karniadakis(mdot: NAL, T: NAL, l: float, h: float,
                              w: float, pin: NAL, pout: NAL, gas: str):
    """Calculate the dimensionless flow according to
    Karniadakis & Beskok: Microflows and Nanoflows: Fundamentals 
    and Simulation,
    p. 144

    Args:
        mdot: Mass flow.
        T: Temperature.
        l, h, w: Channel dimensions.
        pin, pout: Inlet and outlet pressure.
        gas: Gas species.

    Returns:
        Dimensionless mass flow.
    """
    M = ph.get_M(gas)
    R_s = ph.R/M
    p_m = 0.5*(pin + pout)
    Q_dot = (mdot * R_s*T / p_m) / w
    Q_bar = Q_dot * p_m / ((pin-pout)/l * h**2 * (R_s*T)**0.5)
    return Q_bar

def g_to_mdot(mdot_to_g_fun, G, *args):
    """General inverse function to get the mass flow from a given 
    dimensionless mass flow G.
    
    Args:
        mdot_to_g_fun: Function to calculate g from.
        G: Dimensionless mass flow.
        *args: Additional arguments passed to mdot_to_g_fun (everything
            except for mdot).

    Returns:
        Mass flow.    
    """
    G_for_mdot_1 = mdot_to_g_fun(1, *args)
    mdot = G / G_for_mdot_1
    return mdot

def get_transition_diamater(gas: str, ret_var: bool = False):
    """Get the transition diameter of a gas.
    The transition diameter is a diameter related to rarefied gases.
    Kunze, S., Groll, R., Besser, B. et al.
    Molecular diameters of rarefied gases.
    Sci Rep 12, 2057 (2022).
    https://doi.org/10.1038/s41598-022-05871-y

    Args:
        gas: Gas species.
        ret_var: Return the variance of the diameter? Defaults to False.
    
    Returns:
        The transition diameter.
    """
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
