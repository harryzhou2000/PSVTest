import numpy as np
import matplotlib.pyplot as plt

from . import PSV1D


def plot_fourier_1d(elem: PSV1D.PSV1DElem, max_k=np.pi, selectMethod="MinDispErr"):
    ks = np.linspace(0, max_k, 129)
    ks[0] += 1e-5

    reKap = []
    imKap = []
    imKapR = []

    for k in ks:
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
        else:
            reKap.append(kaps[ikapAcc].real)
            imKap.append(kaps[ikapAcc].imag)

        imKapR.append(np.max(np.imag(kaps)))

    y_data = [reKap, imKap, imKapR]
    y_names = ["Re", "Im", "ImR"]

    # Plot each (x, y) data pair in a separate figure window
    for i, (cy, cyname) in enumerate(zip(y_data, y_names)):
        plt.figure(figsize=(6, 4))  # Create a new figure window for each plot
        plt.plot(ks, cy, label=cyname)
        if i == 0:
            plt.plot(ks, ks, label="exact")
        plt.xlabel("kappa")
        plt.ylabel("kappa_num")
        plt.title(cyname)
        plt.legend()  # Show the legend

    plt.show()
