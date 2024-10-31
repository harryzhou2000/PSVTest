import numpy as np


def getPolyOrder():
    dol2D = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [3, 0, 0],
        [2, 1, 0],
        [1, 2, 0],
        [0, 3, 0],
        [4, 0, 0],
        [3, 1, 0],
        [2, 2, 0],
        [1, 3, 0],
        [0, 4, 0],
    ]
    dol3D = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 1, 1],
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
        [3, 1, 0],
        [0, 3, 1],
        [1, 0, 3],
        [1, 3, 0],
        [0, 1, 3],
        [3, 0, 1],
        [2, 2, 0],
        [0, 2, 2],
        [2, 0, 2],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
    ]

    dol2Dn = np.array(dol2D, dtype=np.int32)
    dol3Dn = np.array(dol3D, dtype=np.int32)
    return (dol2Dn, dol3Dn)


def getPolyNum(dim: int, order: int) -> int:
    if dim == 1:
        return order + 1
    if dim == 2:
        return (order + 1) * (order + 2) // 2
    if dim == 3:
        return (order + 1) * (order + 2) * (order + 3) // 6


class TaylorBase:
    def __init__(self, dim: int, order: int):
        self.dim = dim
        self.order = order
        assert dim == 2 or dim == 3 or dim == 1
        self.dol2D, self.dol3D = getPolyOrder()
        self.dol1D = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=np.int32
        )
        self.nBase = getPolyNum(dim, order)
        self._dols = [self.dol1D, self.dol2D, self.dol3D][dim - 1][
            : self.nBase, : self.dim
        ]

    @property
    def dols(self):
        return self._dols

    def __call__(self, x: np.ndarray):
        assert x.shape[0] == self.dim
        # for i in range(self.nBase):
        retL = np.power(
            x.reshape(((1,) + tuple(x.shape))),
            self.dols.reshape(
                (
                    self.nBase,
                    self.dim,
                )
                + (1,) * (len(x.shape) - 1)
            ),
        )
        return np.reshape(np.prod(retL, 1), (self.nBase,) + tuple(x.shape[1:]))


# def TaylorBase(x: np.ndarray, degree: int) -> np.ndarray:
#     dim = x.shape[0]
#     x[0]

if __name__ == "__main__":
    pass
