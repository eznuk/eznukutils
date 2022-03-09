"""
physical constants and other quantities
some conversions
"""

import numpy as np

# ----------------------------------------
# physical constants, source: HCnP, p. 1-1
R = 8.3144598           # universal gas constant [kg m^2 / (s^2 mol K)] 
kB = 1.38065e-23        # boltzmann constant [J/K]

def get_mu(gas, T):
    """Returns the dynamic viscosity of given gas at given temperature
    by linear interpolation.
    Source: HCnP, pp. 6-242f
    
    Parameters
    ----------
    gas : str
        choices are "He", "CO2" and "N2".
    T : float, array_like
        temperature in K between 20 and 100 °C.
    
    Returns
    -------
    visc : float, array_like
       dynamic viscosity of gas at temp T in Pa*s
    """
    temps = [200, 300, 400]  # [K]
    if np.any(T < min(temps)) or np.any(T > max(temps)):
        raise ValueError("Temperature out of range")
        
    if gas == "He":
        viscs = [15.1e-6, 19.9e-6, 24.3e-6]  # [Pa*s]
    elif gas == "CO2":
        viscs = [10.1e-6, 15.0e-6, 19.7e-6]   # [Pa*s]
    elif gas == "N2":
        viscs = [12.9e-6, 17.9e-6, 22.2e-6]   # [Pa*s]
    elif gas == "Ar":
        viscs = [15.9e-6, 22.7e-6, 28.6e-6]   # [Pa*s]
    else:
        raise ValueError("no valid gas given")

    
    visc = np.interp(T, temps, viscs)
    return visc

def get_M(gas):
    """Returns the molar mass of given gas.
    Source: HCnP, pp. 4-4ff
    
    Parameters
    ----------
    gas : str or array of str
        choices are "He", "CO2" and "N2"
    
    Returns
    -------
    M : float
       Molar mass of given gas(es).
    """
    if isinstance(gas, str):
        # single string
        if gas == "He":
            M = 4.002602e-3         # molar mass of helium [kg/mol]
        elif gas == "CO2":
            M = 44.008e-3           # molar mass of carbon dioxide [kg/mol]
        elif gas == "N2":  
            M = 28.014e-3           # molar mass of nitrogen [kg/mol]
        elif gas == "Ar":
            M = 39.948e-3          # [kg/mol]
        else:
            raise ValueError("no valid gas given")
    else:
        # iterable
        M = np.array([get_M(entry) for entry in gas])

    return M

# -------------------------------
# ---- these can be combined ----
#--------------------------------

def kgstosccm(kgs, gas):
    """Converts kg/s to sccm for either He, CO2 or N2.
    
    Parameters
    ----------
    kgs : float, ndarray
        mass flow in kg/s
    gas : str
        choices are "He", "CO2" or "N2"
        
    Returns
    -------
    sccm : float, ndarray
        mass flow in sccm
    """
    
    # mass of one standard cubic centimeter:
    if gas == "He":
        mscc = 0.179e-6    # [kg]
    elif gas == "CO2":
        mscc = 1.809e-6         # [kg]
    elif gas == "N2":
        mscc = 1.251e-6    # [kg]
    else:
        raise ValueError("no valid gas given")
    
    sccm = kgs * 60 / mscc
    return sccm

def sccmtokgs(sccm, gas):
    """Converts sccm to kg/s for either He, CO2 or N2
    
    Parameters
    ----------
    sccm : float, ndarray
        mass flow in sccm
    gas : str
        choices are "He", "CO2" or "N2"
        
    Returns
    -------
    kgs : float, ndarray
        mass flow in kg/s
    """
    
    # mass of one standard cubic centimeter:
    if gas == "He":
        mscc = 0.179e-6    # [kg]
    elif gas == "CO2":
        mscc = 1.809e-6         # [kg]
    elif gas == "N2":
        mscc = 1.251e-6    # [kg]
    else:
        raise ValueError("no valid gas given")
        
    kgs = sccm / 60 * mscc
    return kgs

def mdot_to_pdot(mdot, gas, T, V):
    pdot = mdot / (V / (R / get_M(gas) * T))
    return pdot
