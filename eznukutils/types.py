import numpy as np
import numpy.typing as npt
from typing import Union, List, TypeVar

# like the numpy ArrayLike, but only with numeric types
#NumericArrayLike = Union[int, float, complex, npt.NDArray[np.float_]]
#NumericArrayLike = Union[int, float, complex, npt.NDArray[np.number]]

NumericArray = npt.NDArray[np.float_]

NumericArrayLike = Union[float, np.float_, NumericArray]

TypeVarNumericArrayLike = TypeVar("TypeVarNumericArrayLike",
                                  NumericArray, float)

# either a List of strings or an ndarray of strings
StringIterable = Union[List[str], npt.NDArray[np.str_]]