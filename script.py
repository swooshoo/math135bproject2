import numpy as np
import matplotlib.pyplot as plt

def f(t, x):
    return -4 * x + 1

def adams_bashforth_moulton(t, x, h):
    n = len(t)
    for i in range(3, n - 1):
        x_pred = x[i] + h / 24 * (55 * f(t[i], x[i]) - 59 * f(t[i - 1], x[i - 1]) + 37 * f(t[i - 2], x[i - 2]) - 9 * f(t[i - 3], x[i - 3]))
        x[i + 1] = x[i] + h / 24 * (9 * f(t[i + 1], x_pred) + 19 * f(t[i], x[i]) - 5 * f(t[i - 1], x[i - 1]) + f(t[i - 2], x[i - 2]))
    return x

def runge_kutta(t, x, h):
    n = len(t)
    for i in range(n - 1):
        k1 = h * f(t[i], x[i])
        k2 = h * f(t[i] + h/2, x[i] + k1/2)
        k3 = h * f(t[i] + h/2, x[i] + k2/2)
        k4 = h * f(t[i] + h, x[i] + k3)
        x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

# Exact solution
def exact_solution(t):
    return (1/4) * (1 - np.exp(-4 * t)) 

# Time parameters
T = 5
h_values = [0.5, 0.1, 0.01, 0.001]

for h in h_values:
    num_steps = int(T / h)
    t = np.linspace(0, T, num_steps + 1)
    x_adams = np.zeros(num_steps + 1)
    x_rk = np.zeros(num_steps + 1)

    # Initial condition
    x_adams[0] = 0.5
    x_rk[0] = 0.5

    # Adams-Bashforth-Moulton method
    x_adams = adams_bashforth_moulton(t, x_adams, h)

    # Runge-Kutta method
    x_rk = runge_kutta(t, x_rk, h)

    # Exact solution
    x_exact = exact_solution(t)

    # Plotting
    plt.figure()
    plt.plot(t, x_adams, label="Adams-Bashforth-Moulton")
    plt.plot(t, x_rk, label="Runge-Kutta")
    plt.plot(t, x_exact, label="Exact")
    plt.title(f"Solutions with h = {h}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)

plt.show()
