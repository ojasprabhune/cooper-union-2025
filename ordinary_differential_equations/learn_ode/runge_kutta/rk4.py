# solutions calculated using the fourth-order Runge-Kutta method

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("module://matplotlib-backend-wezterm")


def f(x, t):
    return -(x**3) + math.sin(t)


a = 0.0
b = 10.0
N = 50
h = (b - a) / N

tpoints = np.arange(a, b, h)
xpoints = []
x = 0.0

for t in tpoints:
    xpoints.append(x)
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(x + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(x + k3, t + h)
    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

plt.plot(tpoints, xpoints)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
