import numpy as np
import matplotlib.pyplot as plt

from . import PSV1D


def plot_fourier_1d(
    elem: PSV1D.PSV1DElem,
    max_k=np.pi,
    selectMethod="MinDispErr",
    useContinuousEV=True,
    kThresRatio=0.25,
    nK=129,
    reallyPlot=False,
    titlePrefix = "",
):
    ks = np.linspace(0, max_k, nK)
    ktc = 0.5
    ks[0] = 1e-1 * ktc
    ks[1] = 2e-1 * ktc
    ks[2] = 4e-1 * ktc
    ks[3] = 6e-1 * ktc
    kThres = max_k * kThresRatio

    reKap = []
    imKap = []
    imKapR = []

    for ik, k in enumerate(ks):
        (kaps, evs, v0, A) = elem.testSingleWave(k)
        kaps /= -1j
        if selectMethod == "GEV":
            ikapAcc = np.argmax(np.abs(evs.conj().T @ v0))
        elif selectMethod == "MinDispErr":
            ikapAcc = np.argmin(np.abs(np.real(kaps) - k))
        elif selectMethod == "MaxDiss":
            ikapAcc = np.argmax(np.imag(kaps))
        elif selectMethod == "MinAbsErr":
            ikapAcc = np.argmin(np.abs(kaps - k))
        elif selectMethod == "MinDisp":
            ikapAcc = np.argmin(np.real(kaps))
        testKap = np.mean((A @ v0) / v0) / (-1j)
        if selectMethod == "TestV":
            reKap.append(testKap.real)
            imKap.append(testKap.imag)
        elif useContinuousEV and k > kThres and ik > 0:
            ikapAcc = np.argmax(np.abs(evMaxPrev.conj() @ evs))
            reKap.append(kaps[ikapAcc].real)
            imKap.append(kaps[ikapAcc].imag)
        else:
            reKap.append(kaps[ikapAcc].real)
            imKap.append(kaps[ikapAcc].imag)

        imKapR.append(np.max(np.imag(kaps)))

        # APrev = A
        # evsPrev = evs
        # kapsprev = kaps
        evMaxPrev = evs[:, ikapAcc]
    reKap = np.array(reKap)
    imKap = np.array(imKap)
    imKapR = np.array(imKapR)

    y_data = [reKap, imKap, imKapR]
    y_names = ["Re", "Im", "ImR"]
    if reallyPlot:
        errs = np.sqrt((reKap-ks)**2 + imKap**2)
        print(f"Errors: {errs[0:4]}; ratio [{np.exp2(np.diff(np.log2(errs[0:4])))}]")
        print(f"max dis {imKapR.max()}")

        # Plot each (x, y) data pair in a separate figure window
        for i, (cy, cyname) in enumerate(zip(y_data, y_names)):
            plt.figure(figsize=(6, 4))  # Create a new figure window for each plot
            plt.plot(ks, cy, label=cyname)
            if i == 0:
                plt.plot(ks, ks, label="exact")
            plt.xlabel("kappa")
            plt.ylabel("kappa_num")
            plt.title(titlePrefix + cyname)
            plt.legend()  # Show the legend

        plt.show()

    return (ks, y_data, y_names)
