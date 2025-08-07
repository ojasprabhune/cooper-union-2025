import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("module://matplotlib-backend-wezterm")


fig, ax = plt.subplots(figsize=(5, 5))


def f(x, t):
    return -(x**3) + np.sin(t)


size = 10
num_dx = 100

t = np.linspace(0, size, num_dx)
dt = size / num_dx
x = []
y = 0.0

for i in range(len(t)):
    k1 = dt * f(y, t[i])
    k2 = dt * f(y + k1 / 2, t[i] + dt / 2)
    k3 = dt * f(y + k2 / 2, t[i] + dt / 2)
    k4 = dt * f(y + k3, t[i] + dt)
    y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    x.append(y)

ax.plot(t, x)
ax.set_xlabel("t")
ax.set_ylabel("x(t)")
plt.show()
