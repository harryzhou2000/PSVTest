import numpy as np


class Quad1D:
    def __init__(self, order: int, param_space: str = "Line"):
        assert param_space == "Line"
        self._param_space = param_space
        self._order = order
        
        
    def param_space_volume(self) -> float:
        if self._param_space == "Line":
            return 2.0

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        if self._order <= 1:
            return (
                np.array([[0]], dtype=np.float64),
                np.array([2], dtype=np.float64),
            )
        elif self._order <= 3:
            return (
                np.array([[-0.577350269189626, 0.577350269189626]], dtype=np.float64),
                np.array([1, 1], dtype=np.float64),
            )
        elif self._order <= 5:
            return (
                np.array(
                    [[-0.774596669241483, 0, 0.774596669241483]], dtype=np.float64
                ),
                np.array(
                    [0.555555555555555, 0.888888888888889, 0.555555555555555],
                    dtype=np.float64,
                ),
            )
        elif self._order <= 7:
            return (
                np.array(
                    [
                        [
                            -0.861136311594053,
                            -0.339981043584856,
                            0.339981043584856,
                            0.861136311594053,
                        ]
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454],
                    dtype=np.float64,
                ),
            )
        else:
            raise ValueError(f"no such order [{self._order}] supported")
