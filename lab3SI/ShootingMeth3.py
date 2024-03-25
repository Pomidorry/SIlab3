import numpy as np
import matplotlib.pyplot as plt

def fun1(x, y, z, N, h, t):
    for k in range(N - 1):
        y[k + 1] = y[k] + h * z[k]
        x[k + 1] = x[k] + h * (x[k]**2 - 5 * t[k]**2 - z[k]**2 * np.cos(2 * t[k] + y[k]))
        z[k + 1] = z[k] + h * (4 + y[k]**3 * x[k]**2 + 4 * np.sin(t[k] * (x[k]**2 - 5 * t[k]**2 - z[k]**2 * np.cos(2 * t[k] + z[k]))))
    return x, y, z

def execute_shooting(xa, xb, N, lb, ub):
    h = np.divide((xb - xa), (N - 1))
    t = np.linspace(xa, xb, N)
    range_values = np.arange(lb, ub, 0.1)
    X = np.zeros(N)
    X[0] = 2
    Y = np.zeros(N)
    Y[0] = 2
    Z = np.zeros(N)
    f = np.zeros_like(range_values)
    g = np.zeros_like(range_values)

    for index, value in enumerate(range_values):
        Z[0] = value
        X, Y, Z = fun1(X, Y, Z, N, h, t)
        f[index] = (Y[N-1])**2
        g[index] = abs(Y[N-1])
        plt.plot(X, Y, 'g')
        plt.axis([xa, xb, -5, 5])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y(\alpha)$')
        plt.title(r'$y(\alpha)$')
    plt.show()

    fig2, ax1 = plt.subplots()
    ax1.plot(range_values, f, 'b-')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$J(\alpha)$')
    ax2 = ax1.twinx()
    ax2.plot(range_values, g, 'r-')
    plt.show()

execute_shooting(xa=1, xb=3, N=10, lb=-3, ub=3)
