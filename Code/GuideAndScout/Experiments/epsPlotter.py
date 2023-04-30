from matplotlib import pyplot as plt
import numpy as np

epsEnd = 0.05
epsStart = 0.9
epsDecays = [8000, 10000, 12000, 14000, 16000, 20000]

# epsDecays = [50000, 60000, 70000, 80000]
for epsDecay in epsDecays:
    rec = []
    for eps_done in range(150000):
        epsThresh = epsEnd + (epsStart - epsEnd) * \
            np.exp(-1. * eps_done / epsDecay)
        rec.append(epsThresh)

    plt.plot(rec)

plt.legend(epsDecays, loc="upper left")
plt.show()
