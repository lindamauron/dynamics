import numpy as np
from netket.operator.spin import sigmax, sigmaz
from typing import List
from netket.hilbert import Spin as SpinHilbert


def Hx(hi: SpinHilbert, h: List = None):
    if h is None:
        h = np.ones((hi.size,))
    elif isinstance(h, (int, float)):
        h = h * np.ones((hi.size,))
    return sum([h[i] * sigmax(hi, i) for i in range(hi.size)])


def Hz(hi: SpinHilbert, h: List = None):
    if h is None:
        h = np.ones((hi.size,))
    elif isinstance(h, (int, float)):
        h = h * np.ones((hi.size,))
    return sum([h[i] * sigmaz(hi, i) for i in range(hi.size)])


def Hzz(hi: SpinHilbert, edges: List = None, J: List = None):
    if edges is None:
        edges = [(i, j) for i in range(hi.size - 1) for j in range(i + 1, hi.size)]
    if J is None:
        J = np.ones((len(edges),))
    elif isinstance(J, (int, float)):
        J = J * np.ones((len(edges),))

    return sum([J[k] * sigmaz(hi, i) * sigmaz(hi, j) for k, (i, j) in enumerate(edges)])
