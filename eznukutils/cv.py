"""
Utility functions related to the constant volume method.
All units in SI unless otherwise stated.
"""

import numpy as np

from . import physics as ph
from .types import NumericArrayLike as NAL, StringIterable

def pcovtostd(b__mbar: float, tau: float, pcov: np.ndarray, 
              V: float, T__degrC: float, gas: str) -> float:
    """Calculate the standard deviation of the mass flow resulting from
    an exponential pressure drop fit.
    
    Args:
        b__mbar: fitting parameter 'b__mbar'
        tau: fitting parameter 'tau'
        pcov: array containing the covariance
        V: volume of vessel in m^3
        T__degrC: temperature of vessel in degr C
        gas: name of gas; choices are "He", "CO2" and "N2"
    
    Returns:
        standard deviation of the mass flow balance in sccm
    """
    
    stds = np.sqrt(np.diag(pcov))
    pdot_std__mbar = stds[0]/tau + stds[1]*b__mbar/tau**2
    mdot_balance_std = pdot_std__mbar*100 * V / (ph.R / ph.get_M(gas) 
                                                 * (273.15+np.mean(T__degrC)))
    return ph.kgstosccm(mdot_balance_std, gas)

def calc_pressure_rise_time(p_now__mbar: float, p_goal__mbar: float,
                            mdot__sccm: float, gas: str, T: float,
                            V: float):
    """Prints the time needed to reach a desired vessel 
    pressure with with given mass flow through an MFC.

    Args:
        p_now__mbar: Current pressure im mbar.
        p_goal__mbar: Goal pressure in mbar.
        mdot__sccm: MFC mass flow in sccm.
        gas: Gas species.
        T: Temperature in K.
        V: Volume of the vessel.
    """
    import datetime
    pdot = ph.mdot_to_pdot(ph.sccmtokgs(mdot__sccm, gas), gas, T, V)
    t_ = round((p_goal__mbar-p_now__mbar) * 100 / pdot)
    print(f"Time needed to reach pressure: {str(datetime.timedelta(seconds=t_))} hh:mm:ss")
