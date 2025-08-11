import random
from math import exp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from learn_ode.lotka_volterra.data import freq_fox, freq_rabbit, t

matplotlib.use("module://matplotlib-backend-wezterm")

fig, ax = plt.subplots(figsize=(10, 10))


def prey(alpha, beta, x, y):
    return alpha * x - beta * x * y


def predator(gamma, delta, y, x):
    return -gamma * y + delta * x * y


def energy_prey(alpha, beta, x, y, i):
    f_prey = freq_rabbit[i + 1] - freq_rabbit[i]
    f_hat_prey = prey(alpha, beta, x, y)
    loss_prey = (f_hat_prey - f_prey) ** 2

    return loss_prey


def energy_pred(gamma, delta, x, y, i):
    f_pred = freq_fox[i + 1] - freq_fox[i]
    f_hat_pred = predator(gamma, delta, x, y)
    loss_pred = (f_hat_pred - f_pred) ** 2

    return loss_pred


def perturbation(value: float, decay: float):
    value += random.uniform(-1 * decay, 1 * decay)
    if value <= 0:
        value = 0
    return value


def acceptance_function(T: float, dE: float):
    """
    input:
    t = the temperature
    dE (delta E) = the energy variation between the new candidate solution and current one

    output:
    returns true if the new solution is accepted. otherwise, return false.
    """

    if dE <= 0:
        return True
    else:
        # generate a random value from 0 to 1
        r = random.random()
        if r < exp(-dE / T):
            return True
        else:
            return False


def simulated_annealing_optim(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    x: float,
    y: float,
    T_max_prey: float,
    T_min_prey: float,
    T_max_pred: float,
    T_min_pred: float,
    E_prey_thresh: float,
    E_pred_thresh: float,
    phi: float,
    iteration: int,
    decay: float,
):
    """
    input:
    T_max: the maximum temperature
    T_min: the minimum temperature
    E_thresh: the energy threshold to stop the algorithm
    phi: the cooling factor

    output:
    the best found solution
    """

    T_prey = T_max_prey
    E_prey = energy_prey(alpha, beta, x, y, iteration)

    while T_prey > T_min_prey and E_prey > E_prey_thresh:
        # print("T:", T_prey)
        # print("E:", E_prey)
        # print("alpha:", alpha)
        # print("beta:", beta)
        # print()
        alpha_new = perturbation(alpha, decay)
        beta_new = perturbation(beta, decay)
        E_new_prey = energy_prey(alpha_new, beta_new, x, y, iteration)

        dE_prey = E_new_prey - E_prey

        if acceptance_function(T_prey, dE_prey):
            alpha = alpha_new
            beta = beta_new
            E_prey = E_new_prey

        T_prey *= phi

    T_pred = T_max_pred
    E_pred = energy_pred(gamma, delta, x, y, iteration)

    while T_pred > T_min_pred and E_pred > E_pred_thresh:
        gamma_new = perturbation(gamma, decay)
        delta_new = perturbation(delta, decay)
        E_new_pred = energy_pred(gamma_new, delta_new, x, y, iteration)

        dE_pred = E_new_pred - E_pred

        if acceptance_function(T_pred, dE_pred):
            gamma = gamma_new
            delta = delta_new
            E_pred = E_new_pred

        T_pred *= phi

    # final correct values
    return alpha, beta, gamma, delta


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

epochs = 2500

for epoch in range(epochs):
    for index, t_step in enumerate(t):
        if index == len(t) - 1:
            break
        # iterate through data
        x = freq_rabbit[index]
        y = freq_fox[index]
        alpha, beta, gamma, delta = simulated_annealing_optim(
            alpha,
            beta,
            gamma,
            delta,
            x,
            y,
            T_max_pred=25,
            T_min_pred=1,
            T_max_prey=30,
            T_min_prey=1,
            E_pred_thresh=0.05,
            E_prey_thresh=0.05,
            phi=0.99,
            iteration=index,
            decay=(1 / (epoch + 1)),
        )

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
    "/home/prabhune/projects/cooper-union-2025/ordinary_differential_equations/learn_ode/lotka_volterra/sim_annealing_graphs/1500.png"
)
