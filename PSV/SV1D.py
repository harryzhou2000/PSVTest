import numpy as np
from . import Quad as Quad
from . import Base as Base
from . import PSV1D as PSV1D
import warnings


class SV1DElem(PSV1D.PSV1DElem):
    def __init__(
        self,
        order: int = 3,
        param_space: str = "Line",
        portions=(-0.5, 0, 0.5),
        use_mu=False,
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
        self._use_mu = use_mu

    def _GetSubElems(self):
        self._elems = []
        if len(self._portions) == self.order:
            self._elems.append(np.array([[-1, self._portions[0]]]))
            self._elems.append(np.array([[self._portions[0], self._portions[1]]]))
            self._elems.append(np.array([[self._portions[1], self._portions[2]]]))
            self._elems.append(np.array([[self._portions[2], 1]]))
        elif len(self._portions) == (self.order + 1) * 2:
            for i in range(0, self.order + 1):
                self._elems.append(
                    np.array([[self._portions[i * 2], self._portions[i * 2 + 1]]])
                )
        else:
            raise ValueError("Invalid number of portions")

    def testSingleWave(
        elem, k: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        MInv = np.linalg.pinv(elem.sub_meanF)
        if np.linalg.cond(elem.sub_meanF) > 1e10:
            warnings.warn("Matrix is singular")
        A = np.zeros((elem.nBase, elem.nBase), dtype=np.complex128)

        muInternal = 0.5 * 1 * 2

        MR = elem.BaseF(np.array([[1]])).transpose()[:, :] @ MInv
        ML = elem.BaseF(np.array([[-1]])).transpose()[:, :] @ MInv

        MmuR = 1 * (ML * np.exp(+1j * k) - MR)
        MmuL = 1 * (ML - MR * np.exp(-1j * k))

        for ie in range(elem.nBase):
            e = elem.elems[ie]

            MRc = elem.BaseF(elem.elems[ie][:, 1:2]).transpose()[:, :] @ MInv
            MLc = elem.BaseF(elem.elems[ie][:, 0:1]).transpose()[:, :] @ MInv
            MRcD = elem.BaseD(elem.elems[ie][:, 1:2], (1,)).transpose()[:, :] @ MInv
            MLcD = elem.BaseD(elem.elems[ie][:, 0:1], (1,)).transpose()[:, :] @ MInv
            MmuRc = (
                MmuR * (1 + elem.elems[ie][:, 1]) * 0.5
                + MmuL * (1 - elem.elems[ie][:, 1]) * 0.5
            )
            MmuLc = (
                MmuR * (1 + elem.elems[ie][:, 0]) * 0.5
                + MmuL * (1 - elem.elems[ie][:, 0]) * 0.5
            )

            # if ie == 0 or elem.elems[ie][0, 0] == -1:
            if ie == 0 or e[0, 0] == -1:
                A[ie, :] = -MRc + MR * np.exp(-1j * k)
                if elem._use_mu:
                    A[ie, :] += -MmuRc.reshape(-1)
            elif ie == elem.nBase - 1:
                A[ie, :] = -MRc + MLc
                if elem._use_mu:
                    A[ie, :] += MmuLc.reshape(-1)
            else:
                A[ie, :] = -MRc + MLc
                if elem._use_mu:
                    A[ie, :] += (-MmuRc + MmuLc).reshape(-1)
            A[ie, :] /= elem.sub_vol[ie]

        subInts = []
        for ie in range(elem.nBase):
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

        (evals, evecs) = np.linalg.eig(A)

        return (evals * 2, evecs, subInts, A * 2)
        # print(A)
        eigs = np.linalg.eigvals(A) / k
        print(eigs)
        print(np.min(np.abs(eigs + 1j)))
