import sys, os
import numpy as np
import scipy as sp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import PSV.PSV1D as PSV1D
from PSV.Utils import plot_fourier_1d

if __name__ == "__main__":
    elem = PSV1D.PSV1DElem(3, portions=(2 / 4, 2 / 3))
    testBase = elem.Base(np.array([[0.5]]))
    testBaseAns = np.array([[0, 0.5, 0.25 - 1.0 / 3, 0.125]]).transpose()
    testBaseErr = np.linalg.norm(testBase - testBaseAns) / np.linalg.norm(testBaseAns)
    print(testBaseErr)
    assert testBaseErr < 1e-15

    print("Sub Mean Matrix: ")
    print(elem.sub_mean[:, 1:])
    print("Sub Vol: ")
    print(elem.sub_vol)
    print(sp.linalg.eig(elem.sub_mean[:, 1:]))

    ##
    elem.testSingleWave(np.pi * 0.03 * 0.5)
    plot_fourier_1d(elem, max_k=np.pi * 2, selectMethod="MinDispErr")
