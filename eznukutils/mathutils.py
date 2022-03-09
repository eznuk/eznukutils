import numpy as np

def round_to_significant_figures(x, n=3):
    """
    retuns a float rounded to n significant figures.
    x should be a float.
    """
    try:
        return round(x, n - int(np.floor(np.log10(abs(x)))))
    except ValueError:
        return x

def round_pandas_to_significant_figures(x, n=3):
    """
    returns a dataframe or series rounded to
    n significant digits.
    x should be a pd.Series or a pd.Dataframe.col or .row
    """
    return x.apply(round_to_significant_figures, n=n)