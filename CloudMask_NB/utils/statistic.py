import numpy as np
from typing import Union


def pod(a: Union[int, np.ndarray],
        b: Union[int, np.ndarray],
        c: Union[int, np.ndarray],
        d: Union[int, np.ndarray],
        **kwargs):
    return a / (a + b), d / (c + d)


def far(a: Union[int, np.ndarray],
        b: Union[int, np.ndarray],
        c: Union[int, np.ndarray],
        d: Union[int, np.ndarray],
        **kwargs):
    return b / (a + b), c / (c + d)


def hr(a: Union[int, np.ndarray],
       b: Union[int, np.ndarray],
       c: Union[int, np.ndarray],
       d: Union[int, np.ndarray],
       **kwargs):
    return (a + d) / (a + b + c + d)


def kss(a: Union[int, np.ndarray],
        b: Union[int, np.ndarray],
        c: Union[int, np.ndarray],
        d: Union[int, np.ndarray],
        **kwargs):
    return (a * d - c * b) / ((a + b) * (c + d))
