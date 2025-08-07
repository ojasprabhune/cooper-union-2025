import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("module://matplotlib-backend-wezterm")


def prey(alpha, beta, x, y):
    return alpha * x - beta * x * y


def predator(gamma, delta, y, x):
    return -gamma * y + delta * x * y


def rk4_plot(alpha: float, beta: float, gamma: float, delta: float, iteration: int):
    fig, ax = plt.subplots(figsize=(10, 10))

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

    ax.plot(t, x_prey, label="Prey")
    ax.plot(t, x_predator, label="Predator")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    fig.legend()
    fig.savefig(
        f"~/projects/cooper-union-2025/ordinary-differential-equations/lotka_volterra/graphs/{iteration}.png"
    )


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


if __name__ == "__main__":
    alpha = 0.437
    beta = 0.222
    gamma = 0.549
    delta = 0.134
    rk4_plot(alpha, beta, gamma, delta, 0)
