import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import PSV.Utils as Utils
import PSV.PSV1D as PSV1D
import PSV.SV1D as SV1D

if __name__ == "__main__":

    elems = []
    # elems.append((SV1D.SV1DElem(3, portions=(-0.5, 0, 0.5)), "SV 0.5"))
    # elems.append((SV1D.SV1DElem(3, portions=(-0.7887, 0, 0.7887)), "SV 0.7887"))
    # elems.append((SV1D.SV1DElem(3, portions=(-0.9, 0, 0.9)), "SV 0.9"))

    # for a,b in [(0.5, 0.7), (0.5, 0.5), (0.5, 0.3)]:
    #     elems.append(
    #         (
    #             SV1D.SV1DElem(
    #                 3, portions=(-1, -0, -a - b / 2, -a + b / 2, a - b / 2, a + b / 2, 0, 1)
    #             ),
    #             f"OSV {a} {b}",
    #         )
    #     )

    # for a, b in [(0.5, 0.5), (0.5, 0.25), (0.25, 0.5)]:
    #     elems.append(
    #         (
    #             PSV1D.PSV1DElem(3, portions=(a, b), sub_use_extern=True),
    #             f"DOSV {a} {b}",
    #         )
    #     )

    elems.append((SV1D.SV1DElem(3, portions=(-0.5, 0, 0.5)), "SV 0.5"))
    elems.append((SV1D.SV1DElem(3, portions=(-0.7887, 0, 0.7887)), "SV 0.7887"))
    elems.append((SV1D.SV1DElem(3, portions=(-0.9, 0, 0.9)), "SV 0.9"))
    for a,b in [(0.5, 0.7), (0.5, 0.5), (0.5, 0.3)]:
        elems.append(
            (
                SV1D.SV1DElem(
                    3, portions=(-1, -0, -a - b / 2, -a + b / 2, a - b / 2, a + b / 2, 0, 1)
                ),
                f"OSV {a} {b}",
            )
        )

    fig0 = plt.figure(0, figsize=(8, 6))
    fig1 = plt.figure(1, figsize=(8, 6))
    fig2 = plt.figure(2, figsize=(8, 6))

    for elem, name in elems:
        (ks, y_data, y_names) = Utils.plot_fourier_1d(
            elem,
            np.pi * 4,
        )
        (reKap, imKap, imKapR) = (y_data[0], y_data[1], y_data[2])
        errs = np.sqrt((reKap - ks) ** 2 + imKap**2)
        print(name + ": ")
        print(f"Errors: {errs[0:4]}; ratio [{np.exp2(np.diff(np.log2(errs[0:4])))}]")
        print(f"Max imKapR = {np.max(imKap)}")

        plt.figure(0)
        plt.plot(ks, reKap, label=name)

        plt.figure(1)
        plt.plot(ks, imKap, label=name)

        plt.figure(2)
        plt.plot(ks, imKapR, label=name)

    plt.figure(0)
    plt.xlabel("kappa")
    plt.ylabel("Re(kappa_num)")
    plt.plot(ks, ks, label="Exact", ls=":")
    plt.legend()

    plt.figure(1)
    plt.xlabel("kappa")
    plt.ylabel("Im(kappa_num)")
    plt.legend()

    plt.figure(2)
    plt.xlabel("kappa")
    plt.ylabel("ImMax(kappa_num)")
    plt.legend()

    plt.show()
