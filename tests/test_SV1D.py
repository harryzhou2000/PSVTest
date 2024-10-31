import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import PSV.SV1D as SV1D
from PSV.Utils import plot_fourier_1d

if __name__ == "__main__":
    elem = SV1D.SV1DElem(3, portions=(-0.9, 0, 0.9))
    ##
    ktest = np.pi * 0.3 * 0.5
    eigs = elem.testSingleWave(ktest)
    print(eigs / ktest)
    print(np.min(np.abs(eigs / ktest + 1j)))

    plot_fourier_1d(elem, max_k=np.pi * 2, selectMethod="MinDispErr")
