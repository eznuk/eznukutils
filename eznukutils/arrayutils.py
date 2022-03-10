import numpy as np
import numpy.typing as npt
from typing import Tuple

def convert_to_array(x):
    if not type(x) == np.ndarray:
        if type(x) in [int, float, list, np.float64]:
            return np.array(x)
        else:
            raise ValueError(f"{x} has no valid type. Its type is {type(x)}.")
    else:
        return x

def find_nearest(arr: npt.NDArray[np.number],
                 value: float) -> Tuple[int, float]:
    """Find nearest entry in an array.
    
    Args:
        arr: A 1D array containing the data.
        value: Value to find nearest entry in arr for.
    
    Returns:
        ind, val: Index and value of nearest entry.
    """
    
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()
    return np.where(arr==arr[idx])[0][0], arr[idx]

def filter_peaks(dat_arr: npt.NDArray[np.number], peak_thr: float):
    """Filter out single peaks.
    Changes dat_arr in place!
    
    Args:
        dat_arr: Array of data.
        peak_thr: Threshold for peak detection.

    Returns:
        Modified dat_arr.
    """
    
    for ii, dat in enumerate(dat_arr):
        if ii > 0 and ii < len(dat_arr)-1:
            if dat > dat_arr[ii-1] * peak_thr or \
               dat < dat_arr[ii-1] / peak_thr:
                print("### Filtered peak at ID %i" % ii)
                # interpolate the missing value
                dat_arr[ii] = 0.5*(dat_arr[ii-1] + dat_arr[ii+1])