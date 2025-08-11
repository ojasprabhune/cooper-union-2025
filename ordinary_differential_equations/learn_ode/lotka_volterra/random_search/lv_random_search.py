import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from learn_ode.lotka_volterra.data import freq_fox, freq_rabbit, t

matplotlib.use("module://matplotlib-backend-wezterm")

fig, ax = plt.subplots(figsize=(5, 5))


def prey(alpha, beta, x, y):
    return alpha * x - beta * x * y


def predator(gamma, delta, y, x):
    return -gamma * y + delta * x * y


def loss_prey(alpha, beta, x, y, i):
    f_prey = freq_rabbit[i + 1] - freq_rabbit[i]
    f_hat_prey = prey(alpha, beta, x, y)
    loss_prey = (f_hat_prey - f_prey) ** 2

    return loss_prey


def loss_pred(gamma, delta, x, y, i):
    f_pred = freq_fox[i + 1] - freq_fox[i]
    f_hat_pred = predator(gamma, delta, x, y)
    loss_pred = (f_hat_pred - f_pred) ** 2

    return loss_pred


def perturbation(value: float, decay: float):
    value += random.uniform(-1 * decay, 1 * decay)
    value = random.uniform(0, 1)
    return value


def rk4(alpha: float, beta: float, gamma: float, delta: float):
    domain = 20
    num_dx = 10000

    t = np.linspace(0, domain / 2, num_dx)
    dt = domain / num_dx
    x_prey = []
    y_prey = 2
    x_predator = []
    y_predator = 4.4678

    for _ in range(len(t)):
        k1 = dt * prey(alpha, beta, y_prey, y_predator)
        k2 = dt * prey(alpha, beta, y_prey + k1 / 2, y_predator + dt / 2)
        k3 = dt * prey(alpha, beta, y_prey + k2 / 2, y_predator + dt / 2)
        k4 = dt * prey(alpha, beta, y_prey + k3, y_predator + dt)
        y_prey += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_prey.append(y_prey)

        pk1 = dt * predator(gamma, delta, y_predator, y_prey)
        pk2 = dt * predator(gamma, delta, y_predator + pk1 / 2, y_prey + dt / 2)
        pk3 = dt * predator(gamma, delta, y_predator + pk2 / 2, y_prey + dt / 2)
        pk4 = dt * predator(gamma, delta, y_predator + pk3, y_prey + dt)
        y_predator += (pk1 + 2 * pk2 + 2 * pk3 + pk4) / 6
        x_predator.append(y_predator)

    return x_prey, x_predator, t


alpha = random.random()
beta = random.random()
gamma = random.random()
delta = random.random()

start_alpha = alpha
start_beta = beta
start_gamma = gamma
start_delta = delta

epochs = 10000
decay = 0.0000001

for epoch in range(epochs):
    for index, t_step in enumerate(t):
        if index == len(t) - 1:
            break
        # iterate through data
        x = freq_rabbit[index]
        y = freq_fox[index]

        alpha_new = perturbation(alpha, decay)
        beta_new = perturbation(beta, decay)
        gamma_new = perturbation(gamma, decay)
        delta_new = perturbation(delta, decay)

        prey_loss_new = loss_prey(alpha_new, beta_new, x, y, index)
        prey_loss = loss_prey(alpha, beta, x, y, index)

        pred_loss_new = loss_pred(gamma_new, delta_new, x, y, index)
        pred_loss = loss_pred(gamma, delta, x, y, index)

        # print("new prey loss:", prey_loss_new)
        # print("prey loss:", prey_loss)

        if prey_loss_new < prey_loss:
            alpha = alpha_new
            beta = beta_new

        if pred_loss_new < pred_loss:
            gamma = gamma_new
            delta = delta_new

        # print(prey_loss)

    # print(f"Epoch: {epoch}")

print("\nfinal values:")
print("start_alpha:", start_alpha)
print("start_beta:", start_beta)
print("start_gamma:", start_gamma)
print("start_delta:", start_delta)
print()

print("alpha:", alpha)
print("beta:", beta)
print("gamma:", gamma)
print("delta:", delta)

x_prey, x_pred, t_rk4 = rk4(alpha, beta, gamma, delta)
plt.plot(t_rk4, x_prey, color="red")
plt.plot(t_rk4, x_pred, color="blue")
plt.plot(t, freq_rabbit, color="red")
plt.plot(t, freq_fox, color="blue")
plt.show()
fig.savefig(
    f"/home/prabhune/projects/cooper-union-2025/ordinary_differential_equations/learn_ode/lotka_volterra/random_search_graphs/{epochs}.png"
)
