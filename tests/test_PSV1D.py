import sys, os
import numpy as np
import scipy as sp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import PSV.PSV1D as PSV1D

if __name__ == "__main__":
    elem = PSV1D.PSV1DElem(3, portions=(2 / 4, 2 / 4))
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
    def testSingleWave(k: float):
        MInv = np.linalg.pinv(elem.sub_mean[:, 1:])
        A = np.zeros((4, 4), dtype=np.complex128)
        ### Row for um0
        MR = elem.Base(np.array([[1]])).transpose()[:, 1:] @ MInv
        ML = elem.Base(np.array([[-1]])).transpose()[:, 1:] @ MInv
        # print(MR)
        A[0, 0] = -1 + np.exp(-1j * k * 2)
        A[0, 1:] = -MR + MR * np.exp(-1j * k * 2)
        A[0, :] /= 2  # volume
        ### Row for um1
        MR1 = elem.Base(elem.elems[0][:, 1:2]).transpose()[:, 1:] @ MInv
        # print(MR1)
        A[1, 0] = -1 + np.exp(-1j * k * 2)
        A[1, 1:] = -MR1 + MR * np.exp(-1j * k * 2)
        A[1, :] /= elem.sub_vol[0]
        A[1, :] -= A[0, :] # here the internal dofs are sub_mean - mean
        ### Row for um2
        MR2 = elem.Base(elem.elems[1][:, 1:2]).transpose()[:, 1:] @ MInv
        ML2 = elem.Base(elem.elems[1][:, 0:1]).transpose()[:, 1:] @ MInv
        A[2, 1:] = -MR2 + ML2
        A[2, :] /= elem.sub_vol[1]
        A[2, :] -= A[0, :]
        ### Row for um3
        MR3 = elem.Base(elem.elems[2][:, 1:2]).transpose()[:, 1:] @ MInv
        ML3 = elem.Base(elem.elems[2][:, 0:1]).transpose()[:, 1:] @ MInv
        A[3, 1:] = -MR3 + ML3
        A[3, :] /= elem.sub_vol[2]
        A[3, :] -= A[0, :]
        # print(A)
        eigs = np.linalg.eigvals(A)/k
        print(eigs)
        print(np.min(np.abs(eigs + 1j)))

    testSingleWave(np.pi * 0.03 * 0.5)
