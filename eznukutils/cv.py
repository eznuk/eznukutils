"""
helper functions related to the constant volume method
"""

import numpy as np

from . import physics as ph

def pcovtostd(b__mbar, tau, pcov, V, T__degrC, gas):
    """ Calculate the standard deviation of the mass flow resulting from
    an exponential pressure drop fit.
    
    Parameters
    ----------
    b__mbar : float
        fitting parameter 'b__mbar'
    tau : float
        fitting parameter 'tau'
    pcov : ndarray
        array containing the covariance
    V : float
        volume of vessel in m^3
    T__degrC : float
        temperature of vessel in degr C
    gas : string
        name of gas; choices are "He", "CO2" and "N2"
    
    Returns
    -------
    mdot_balance_std__sccm : float
        standard deviation of the mass flow balance in sccm
    """
    
    stds = np.sqrt(np.diag(pcov))
    pdot_std__mbar = stds[0]/tau + stds[1]*b__mbar/tau**2
    mdot_balance_std = pdot_std__mbar*100 * V / (ph.R / ph.get_M(gas) 
                                                 * (273.15+np.mean(T__degrC)))
    return ph.kgstosccm(mdot_balance_std, gas)

def calc_pressure_rise_time(p_now__mbar, p_goal__mbar, mdot__sccm, gas, T):
    """
    prints the time needed to reach a desired DD 
    pressure with with given mass flow through an MFC
    T in K
    """
    import datetime
    pdot = ph.mdot_to_pdot(ph.sccmtokgs(mdot__sccm, gas), gas, T)
    t_ = round((p_goal__mbar-p_now__mbar) * 100 / pdot)
    print(f"Time needed to reach pressure: {str(datetime.timedelta(seconds=t_))} hh:mm:ss")
