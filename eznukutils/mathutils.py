import numpy as np
import pandas as pd
from typing import Union

def round_to_significant_figures(x: float, n: int = 3):
    """Retun a float rounded to n significant figures.
    
    Args:
        x: Number to round.
        n: Number of significiant figures.

    Returns:
        Rounded x.
    """
    try:
        return round(x, n - int(np.floor(np.log10(abs(x)))))
    except ValueError:
        return x

def round_series_to_significant_figures(
        x: pd.Series, n: int = 3) -> pd.Series:
    """Return a dataframe or series rounded to
    n significant digits.
    
    Args:
        x: Numbers to round.
        n: Number of significiant figures.
    
    Returns:
        Rounded x.
    """
    return x.apply(round_to_significant_figures, n=n)