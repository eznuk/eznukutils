"""
Physical constants and other quantities.
Some conversions.
All units in SI unless otherwise stated.
"""

import numpy as np
from typing import Union, overload

from .types import (NumericArrayLike as NAL,
                    TypeVarNumericArrayLike as TNAL,
                    NumericArray as NA,
                    StringIterable)

# ----------------------------------------
# physical constants, source: HCnP, p. 1-1
R = 8.3144598           # universal gas constant [kg m^2 / (s^2 mol K)] 
kB = 1.38065e-23        # boltzmann constant [J/K]

def get_mu(gas: str, T: NAL) -> NAL:
    """Returns the dynamic viscosity of given gas at given temperature
    by linear interpolation.
    Source: HCnP, pp. 6-242f
    
    Args:
        gas: choices are "He", "CO2" and "N2".
        T: temperature in K between 20 and 100 °C.
    
    Returns:
        visc: dynamic viscosity of gas at temp T in Pa*s
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

@overload
def get_M(gas: str) -> float:
    ...

@overload
def get_M(gas: StringIterable) -> NA:
    ...

def get_M(gas):
    """Returns the molar mass of given gas.
    Source: HCnP, pp. 4-4ff
    
    Args:
        gas: str or array of str
        choices are "He", "CO2" and "N2"
    
    Returns:
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

def get_mscc(gas: str) -> float:
    """mass of one standard cubic centimeter of a certain gas."""

    if gas == "He":
        return 0.179e-6    # [kg]
    elif gas == "CO2":
        return 1.809e-6         # [kg]
    elif gas == "N2":
        return 1.251e-6    # [kg]
    else:
        raise ValueError("no valid gas given")

def kgstosccm(kgs: TNAL, gas: str) -> TNAL:
    """Converts kg/s to sccm for either He, CO2 or N2.
    
    Args:
        kgs: mass flow in kg/s
        gas: choices are "He", "CO2" or "N2"
        
    Returns:
        mass flow in sccm
    """
    
    # mass of one standard cubic centimeter:
    mscc = get_mscc(gas)
    
    sccm = kgs * 60 / mscc
    return sccm

def sccmtokgs(sccm: TNAL, gas: str) -> TNAL:
    """Converts sccm to kg/s for either He, CO2 or N2
    
    Args:
        sccm: mass flow in sccm
        gas: choices are "He", "CO2" or "N2"
        
    Returns:
        kgs: mass flow in kg/s
    """
    
    # mass of one standard cubic centimeter:
    mscc = get_mscc(gas)
        
    kgs = sccm / 60 * mscc
    return kgs

def mdot_to_pdot(mdot: TNAL, gas: str, T: TNAL, V: TNAL) -> TNAL:
    pdot = mdot / (V / (R / get_M(gas) * T))
    return pdot
