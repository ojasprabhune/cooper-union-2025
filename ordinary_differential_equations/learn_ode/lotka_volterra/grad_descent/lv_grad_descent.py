import random

import matplotlib
import matplotlib.pyplot as plt
from learn_ode.lotka_volterra.data import freq_fox, freq_rabbit, t
from runge_kutta import rk4_plot, rk4

matplotlib.use("module://matplotlib-backend-wezterm")

fig, ax = plt.subplots(figsize=(10, 10))


def prey(alpha, beta, x, y):
    return alpha * x - beta * x * y


def predator(gamma, delta, y, x):
    return -gamma * y + delta * x * y


alpha = random.random()
beta = random.random()
gamma = random.random()
delta = random.random()

start_alpha = alpha
start_beta = beta
start_gamma = gamma
start_delta = delta

lr = 0.001
epochs = 1000

for epoch in range(epochs):
    for i, t_step in enumerate(t):
        if i == len(t) - 1:
            break
        x = freq_rabbit[i]
        y = freq_fox[i]
        f_prey = freq_rabbit[i + 1] - freq_rabbit[i]
        f_pred = freq_fox[i + 1] - freq_fox[i]
        f_hat_prey = prey(alpha, beta, x, y)
        f_hat_pred = predator(gamma, delta, x, y)
        loss_prey = (f_hat_prey - f_prey) ** 2
        loss_pred = (f_hat_pred - f_pred) ** 2
        print("prey loss:", loss_prey)
        print("pred loss:", loss_pred)
        alpha -= lr * (2 * alpha * x**2 - 2 * beta * x**2 * y - 2 * f_prey * y)
        beta -= lr * (
            -2 * alpha * x**2 * y + 2 * beta * x**2 * y**2 + 2 * f_prey * x * y
        )

        gamma -= lr * (2 * gamma * y**2 - 2 * delta * y**2 * x + 2 * f_pred * y)
        delta -= lr * (
            -2 * gamma * y**2 * x + 2 * delta * x**2 * y**2 - 2 * f_pred * x * y
        )

    if epoch % 200 == 0:
        rk4_plot(alpha, beta, gamma, delta, epoch)

        print("epoch:", epoch)
        print("alpha:", alpha)
        print("beta:", beta)
        print("gamma:", gamma)
        print("delta:", delta)
        print()

rk4_plot(alpha, beta, gamma, delta, epochs)

print("\nfinal values:")
print("start_alpha:", start_alpha)
print("start_beta:", start_beta)
print("start_gamma:", start_gamma)
print("start_delta:", start_delta)

print("alpha:", alpha)
print("beta:", beta)
print("gamma:", gamma)
print("delta:", delta)

x_prey, x_pred, t_rk4 = rk4(alpha, beta, gamma, delta)
ax.plot(t_rk4, x_prey, label="Simulated rabbits", color="red")
ax.plot(t, freq_rabbit, label="Data rabbits", color="blue")
ax.set_xlabel("Time")
ax.set_ylabel("Frequency")
fig.legend()
fig.savefig(
    "/home/prabhune/projects/cooper-union-2025/ordinary_differential_equations/learn_ode/lotka_volterra/grad_descent_graphs/rabbits.png"
)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(t_rk4, x_pred, label="Simulated foxes", color="red")
ax.plot(t, freq_fox, label="Data foxes", color="blue")
ax.set_xlabel("Time")
ax.set_ylabel("Frequency")
fig.legend()
fig.savefig(
    "/home/prabhune/projects/cooper-union-2025/ordinary_differential_equations/learn_ode/lotka_volterra/grad_descent_graphs/foxes.png"
)
