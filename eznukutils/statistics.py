import numpy as np

def running_mean(x, N):
    """calculates the running mean of a signal x with window length N"""
    if N % 2: N += 1  # make argument even
    # insert values before and after data to get a result which
    # equals x in length
    augmented = np.insert(x, 0, np.ones(int(N/2))*x[0])
    augmented = np.insert(augmented, -1, np.ones(int(N/2))*x[-1])
    # do the running mean the fancy way
    # (taken from https://stackoverflow.com/a/27681394)
    cumsum = np.cumsum(augmented)
    return (cumsum[N:] - cumsum[:-N]) / N

def r_squared(dat, fit):
    """Calculating R^2 of a fit.
    
    Parameters
    ----------
    dat : ndarray
        original data
    fit : ndarray
        fitted data
    
    Returns
    -------
    r_sq : float
        R^2 value
    """
    residuals = dat - fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((dat-np.mean(dat))**2)
    r_sq = 1 - (ss_res / ss_tot)
    return r_sq