# solutions calculated using the second-order Runge-Kutta method

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("module://matplotlib-backend-wezterm")


def f(x, t):
    return -(x**3) + math.sin(t)


a = 0.0
b = 10.0
N = 100
h = (b - a) / N

tpoints = np.arange(a, b, h)
xpoints = []

x = 0.0
for t in tpoints:
    xpoints.append(x)
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
    x += k2

plt.plot(tpoints, xpoints)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
