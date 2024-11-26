import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PSV.Base import *

if __name__ == "__main__":
    assert getPolyNum(1, 4) == 5
    assert getPolyNum(2, 3) == 10
    assert getPolyNum(3, 3) == 20

    points = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 0.5],
            [1, 0],
            [1, 1],
            [1, 0.5],
            [1, 0],
            [1, 1],
            [1, 0.5],
        ]
    )

    points = np.transpose(points, axes=(1, 0))
    points = points.reshape((2, 3, 3))
    base = TaylorBase(dim=2, order=3)

    bV = base(points)
    dbV = base(points, diff=(1, 0))
    assert bV.shape == (10, 3, 3) and dbV.shape == (10, 3, 3)
    
    print(points[:, 0, 2])
    print(bV[:, 0, 2])
    print(dbV[:, 0, 2])
    
    print(points[:, 1, 1])
    print(bV[:, 1, 1])
    print(dbV[:, 1, 1])
