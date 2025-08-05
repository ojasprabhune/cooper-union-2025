# solution of a differential equation to t = infinity

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("module://matplotlib-backend-wezterm")


def g(x, u):
    return 1 / (x**2 * (1 - u) ** 2 + u**2)


a = 0.0
b = 1.0
N = 100
h = (b - a) / N
upoints = np.arange(a, b, h)
tpoints = []
xpoints = []
x = 1.0
for u in upoints:
    tpoints.append(u / (1 - u))
    xpoints.append(x)
    k1 = h * g(x, u)
    k2 = h * g(x + 0.5 * k1, u + 0.5 * h)
    k3 = h * g(x + 0.5 * k2, u + 0.5 * h)
    k4 = h * g(x + k3, u + h)
    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

print(len(tpoints))
plt.plot(tpoints, xpoints)
plt.xlim(0, 80)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
