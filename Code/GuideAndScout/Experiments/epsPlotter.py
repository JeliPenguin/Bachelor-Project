from matplotlib import pyplot as plt
import numpy as np

epsEnd = 0.05
epsStart = 0.9
epsDecay = 12000
rec = []


for eps_done in range(100000):
    epsThresh = epsEnd + (epsStart - epsEnd) * \
        np.exp(-1. * eps_done / epsDecay)
    rec.append(epsThresh)

plt.plot(rec)
plt.show()
