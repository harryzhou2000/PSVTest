import numpy as np
from . import Quad as Quad
from . import Base as Base
from . import PSV1D as PSV1D


class SV1DElem(PSV1D.PSV1DElem):
    def __init__(
        self, order: int = 3, param_space: str = "Line", portions=(-0.5, 0, 0.5)
    ):
        """ """
        assert param_space == "Line"
        assert order in [3]
        self._order = order
        self._int_order = order * 2
        quadHelper = Quad.Quad1D(order=self._int_order)
        self._quad = quadHelper.points()
        self._base = Base.TaylorBase(1, order=self._order)
        self._GetMean()

        self._portions = portions  # use (1./3, 1./3) for filled PSV
        self._GetSubElems()
        self._GetSubMean()

    def _GetSubElems(self):
        self._elems = []
        self._elems.append(np.array([[-1, self._portions[0]]]))
        self._elems.append(np.array([[self._portions[0], self._portions[1]]]))
        self._elems.append(np.array([[self._portions[1], self._portions[2]]]))
        self._elems.append(np.array([[self._portions[2], 1]]))

    def testSingleWave(elem, k: float):
        MInv = np.linalg.pinv(elem.sub_meanF)
        A = np.zeros((elem.nBase, elem.nBase), dtype=np.complex128)
        for ie in range(elem.nBase):
            e = elem.elems[ie]
            MR = elem.BaseF(np.array([[1]])).transpose()[:, :] @ MInv
            ML = elem.BaseF(np.array([[-1]])).transpose()[:, :] @ MInv
            MRc = elem.BaseF(elem.elems[ie][:, 1:2]).transpose()[:, :] @ MInv
            MLc = elem.BaseF(elem.elems[ie][:, 0:1]).transpose()[:, :] @ MInv
            if ie == 0:
                A[ie, :] = -MRc + MR * np.exp(-1j * k * 2)
            elif ie == elem.nBase - 1:
                A[ie, :] = -MRc + MLc
            else:
                A[ie, :] = -MRc + MLc
            A[ie, :] /= elem.sub_vol[ie]

        return np.linalg.eigvals(A)
        # print(A)
        eigs = np.linalg.eigvals(A) / k
        print(eigs)
        print(np.min(np.abs(eigs + 1j)))
