import numpy as np

def convert_to_array(x):
    if not type(x) == np.ndarray:
        if type(x) in [int, float, list, np.float64]:
            return np.array(x)
        else:
            raise ValueError(f"{x} has no valid type. Its type is {type(x)}.")
    else:
        return x

def find_nearest(array, value):
    """finds nearest entry in an array.
    
    Parameters
    ----------
    array : array_like
        data
    value : float
        value to find nearest entry in data for
    
    Returns
    -------
    ind, val : float
        index and value of nearest entry
    """
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return np.where(array==array[idx])[0][0], array[idx]

def filter_peaks(dat_arr, peak_thr):
    """Filters out single peaks.
    Changes dat_arr in place!
    
    Parameters
    ----------
    dat_arr : array_like
        array of data
    peak_thr : float
        threshold for peak detection.
    """
    
    for ii, dat in enumerate(dat_arr):
        if ii > 0 and ii < len(dat_arr)-1:
            if dat > dat_arr[ii-1] * peak_thr or \
               dat < dat_arr[ii-1] / peak_thr:
                print("### Filtered peak at ID %i" % ii)
                # interpolate the missing value
                dat_arr[ii] = 0.5*(dat_arr[ii-1] + dat_arr[ii+1])