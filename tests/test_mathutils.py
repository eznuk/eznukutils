import pandas as pd

from eznukutils import mathutils

def test_round_pandas_to_significant_figures():
    series = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": [1.2334, 2.343443, 3.1],
                       "b": [2, 3.5634234234, 4.1]})

    ret_series = mathutils.round_series_to_significant_figures(series)
    assert isinstance(ret_series, pd.Series)

    ret_series = mathutils.round_series_to_significant_figures(df["a"])
    assert isinstance(ret_series, pd.Series)