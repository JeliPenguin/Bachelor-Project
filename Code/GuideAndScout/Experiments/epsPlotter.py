from matplotlib import pyplot as plt
import numpy as np

epsEnd = 0.05
epsStart = 0.9
epsDecays = [10000, 12000, 14000, 16000, 20000]


for epsDecay in epsDecays:
    rec = []
    for eps_done in range(100000):
        epsThresh = epsEnd + (epsStart - epsEnd) * \
            np.exp(-1. * eps_done / epsDecay)
        rec.append(epsThresh)

    plt.plot(rec)

plt.legend(epsDecays,loc="upper left")
plt.show()
