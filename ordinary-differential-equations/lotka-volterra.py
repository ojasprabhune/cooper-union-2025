import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("module://matplotlib-backend-wezterm")

fig, ax = plt.subplots(figsize=(5, 5))

alpha = 1
beta = 1
gamma = 1
delta = 1


def prey(x, y):
    return alpha * x - beta * x * y


def predator(y, x):
    return -gamma * y + delta * x * y


domain = 20
num_dx = 10000

t = np.linspace(-(domain / 2), domain / 2, num_dx)
dt = domain / num_dx
x_prey = []
y_prey = 1.0
x_predator = []
y_predator = 1.0

for i in range(len(t)):
    k1 = dt * prey(y_prey, y_predator)
    k2 = dt * prey(y_prey + k1 / 2, y_predator + dt / 2)
    k3 = dt * prey(y_prey + k2 / 2, y_predator + dt / 2)
    k4 = dt * prey(y_prey + k3, y_predator + dt)
    y_prey += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    x_prey.append(y_prey)

    pk1 = dt * predator(y_predator, y_prey)
    pk2 = dt * predator(y_predator + pk1 / 2, y_prey + dt / 2)
    pk3 = dt * predator(y_predator + pk2 / 2, y_prey + dt / 2)
    pk4 = dt * predator(y_predator + pk3, y_prey + dt)
    y_predator += (pk1 + 2 * pk2 + 2 * pk3 + pk4) / 6
    x_predator.append(y_predator)

ax.plot(t, x_prey, label="Prey")
ax.plot(t, x_predator, label="Predator")
ax.set_xlabel("t")
ax.set_ylabel("x(t)")
fig.legend()
plt.show()
