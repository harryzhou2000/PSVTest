import numpy as np
from . import Quad as Quad
from . import Base as Base


class PSV1DElem:
    def __init__(
        self, order: int = 3, param_space: str = "Line", portions=(1.0 / 6, 1.0 / 3)
    ):
        """ """
        assert param_space == "Line"
        assert order in [3]
        self._order = order
        self._int_order = order * 2
        quadHelper = Quad.Quad1D(order=self._int_order)
        self._quad = quadHelper.points()
        self._base = Base.TaylorBase(1, order=self._order)
        self.__GetMean()

        self._portions = portions  # use (1./3, 1./3) for filled PSV
        self.__GetSubElems()
        self.__GetSubMean()

    def __GetMean(self):
        self._mean = self._base(self._quad[0]) @ self._quad[1]
        self._quad_V = self._mean[0]
        self._mean /= self._quad_V  # first one must be the 1 base

    def __GetSubElems(self):
        self._elems = []
        self._elems.append(np.array([[-1, -1 + self._portions[1]]]))
        self._elems.append(np.array([[1 - self._portions[1], 1]]))
        self._elems.append(
            np.array([[-self._portions[0] / 2 * 0, self._portions[0] / 2]])
        )

    def __GetSubMean(self):
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
    def sub_vol(self):
        return self._sub_vol
