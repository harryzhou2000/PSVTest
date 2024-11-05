import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import PSV.SV1D as SV1D
from PSV.Utils import plot_fourier_1d

if __name__ == "__main__":
    elem = SV1D.SV1DElem(3, portions=(-0.8, 0, 0.8))
    # elem = SV1D.SV1DElem(3, portions=(-1, -0.5, -1, 0, 0, 1, 0.5, 1))
    sep = 0.3
    wid = 0.4
    # elem = SV1D.SV1DElem(
    #     3, portions=(-1, -0, -wid - sep, 0 - sep, 0 + sep, wid + sep, 0, 1)
    # )
    ##
    ktest = np.pi * 0.03 * 0.5
    (eigs, evs, v0, A) = elem.testSingleWave(ktest)
    print(eigs / ktest)
    print(np.min(np.abs(eigs / ktest + 1j)))
    print("EVS:")
    print(evs.shape)
    print(evs.transpose())
    print("V0:")
    print(v0)
    print(np.abs(evs.conj().T @ v0))
    print(np.diagonal(evs.conj().T @ evs))
    plot_fourier_1d(
        elem,
        max_k=np.pi * 4,
        # selectMethod="TestV",
        # selectMethod="MaxDiss",
        reallyPlot=True,
    )
