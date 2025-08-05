# numerical solution of an ordinary differential equation
# a solution to dx/dt = -x^3 + sin t from x = 0 to x = 10,
# calculated using Euler's method

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("module://matplotlib-backend-wezterm")


def f(x, t):
    return -(x**3) + math.sin(t)


a = 0.0
b = 10.0
N = 1000
h = (b - a) / N
x = 0.0

tpoints = np.arange(a, b, h)
xpoints = []

for t in tpoints:
    xpoints.append(x)
    x += h * f(x, t)

plt.plot(tpoints, xpoints)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
