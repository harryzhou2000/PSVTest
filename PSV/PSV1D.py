import numpy as np
from . import Quad as Quad
from . import Base as Base
import warnings


class PSV1DElem:
    def __init__(
        self,
        order: int = 3,
        param_space: str = "Line",
        portions=(1.0 / 6, 1.0 / 3),
        sub_use_extern=True,
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

        self._sub_use_extern = sub_use_extern

    def _GetMean(self):
        self._mean = self._base(self._quad[0]) @ self._quad[1]
        self._quad_V = self._mean[0]
        self._mean /= self._quad_V  # first one must be the 1 base

    def _GetSubElems(self):
        self._elems = []
        self._elems.append(np.array([[-1, -self._portions[0]]]))
        self._elems.append(np.array([[self._portions[0], 1]]))
        # self._elems.append( # singular if put at center
        #     np.array([[-self._portions[0] / 2, self._portions[0] / 2]])
        # )

        self._elems.append(np.array([[-self._portions[1], 0]]))  # put at left

    def _GetSubMean(self):
        self._sub_mean = []
        for e in self._elems:
            pMain = (
                0.5 * (1 + self._quad[0]) * e[:, 0]
                + 0.5 * (1 - self._quad[0]) * e[:, 1]
            )
            self._sub_mean.append(self.Base(pMain) @ self._quad[1])
        self._sub_mean = np.array(self._sub_mean)
        self._sub_mean /= self._quad_V
        self._sub_vol = []
        for e in self._elems:
            self._sub_vol.append(e[0, 1] - e[0, 0])

    def Base(self, xi: np.ndarray):
        return self._base(xi) - self._mean.reshape((-1,) + (1,) * (len(xi.shape) - 1))

    def BaseF(self, xi: np.ndarray):
        return self._base(xi)

    def BaseD(self, xi: np.ndarray, diff=(0,)):
        return self._base(xi, diff)

    @property
    def order(self) -> int:
        return self._order

    @property
    def int_order(self) -> int:
        return self._int_order

    @property
    def mean(self):
        return self._mean

    @property
    def elems(self):
        return self._elems

    @property
    def sub_mean(self):
        return self._sub_mean

    @property
    def sub_meanF(self):
        return self._sub_mean + np.reshape(self._mean, (1, -1))

    @property
    def sub_vol(self):
        return self._sub_vol

    @property
    def sub_use_extern(self):
        return self._sub_use_extern

    @property
    def nBase(self) -> int:
        return self._base.nBase

    @property
    def nElem(self) -> int:
        return len(self._elems)

    def testSingleWave(
        elem, k: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        MInv = np.linalg.pinv(elem.sub_mean[:, 1:])
        if np.linalg.cond(elem.sub_mean[:, 1:]) > 1e10:
            warnings.warn("Matrix is singular")
        A = np.zeros((4, 4), dtype=np.complex128)
        ### Row for um0
        MR = elem.Base(np.array([[1]])).transpose()[:, 1:] @ MInv
        ML = elem.Base(np.array([[-1]])).transpose()[:, 1:] @ MInv
        # print(MR)
        A[0, 0] = -1 + np.exp(-1j * k)
        A[0, 1:] = -MR + MR * np.exp(-1j * k)
        A[0, :] /= 2  # volume
        ### Row for um1
        MR1 = elem.Base(elem.elems[0][:, 1:2]).transpose()[:, 1:] @ MInv
        ML1 = elem.Base(elem.elems[0][:, 0:1]).transpose()[:, 1:] @ MInv
        # print(MR1)
        if elem.sub_use_extern:
            A[1, 0] = -1 + np.exp(-1j * k)
            A[1, 1:] = -MR1 + MR * np.exp(-1j * k)
        else:
            A[1, 1:] = -MR1 + ML1
        A[1, :] /= elem.sub_vol[0]
        A[1, :] -= A[0, :]  # here the internal dofs are sub_mean - mean
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

        (evals, evecs) = np.linalg.eig(A)
        subInts = []
        subInts.append(
            (
                np.exp(1j * k / 2 * 1)
                - np.exp(1j * k / 2 * -1)  # note 'k' is wavenumber * delta_x
            )
            / 2
        )
        for ie in range(elem.nBase - 1):
            subInts.append(
                (
                    np.exp(1j * k / 2 * elem.elems[ie][0, 1])
                    - np.exp(
                        1j * k / 2 * elem.elems[ie][0, 0]
                    )  # note 'k' is wavenumber * delta_x
                )
                / elem.sub_vol[ie]
            )
        subInts = np.array(subInts)

        return (evals * 2, evecs, subInts, A * 2)

        # return np.linalg.eigvals(A)
        eigs = np.linalg.eigvals(A) / k
        print(eigs)
        print(np.min(np.abs(eigs + 1j)))
