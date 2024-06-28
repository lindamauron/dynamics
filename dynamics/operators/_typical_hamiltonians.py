
from netket.operator.spin import sigmax, sigmaz
from typing import List
from netket.hilbert import Spin as SpinHilbert

def Hx(hi:SpinHilbert, N:int):
    return sum([sigmax(hi,i) for i in range(N)])


def Hzz(hi:SpinHilbert, edges:List):
    return sum([sigmaz(hi,i)*sigmaz(hi,j) for i,j in edges])