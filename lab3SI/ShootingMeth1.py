import numpy as np
import matplotlib.pyplot as plt

def fun1(x, y, z, N, h, t):
    for k in range(N - 1):
        x[k + 1] = x[k] + h * z[k]
        y[k + 1] = y[k] + h * (y[k] ** 4 + x[k] ** 3 - 3 * np.sin(t[k] ** 2))
        if y[k + 1] < -5 or y[k + 1] > 5: 
            y[k + 1] = np.nan
        z[k + 1] = z[k] + h * (x[k] ** 2 + t[k] ** 2 - y[k] ** 2 * np.cos(z[k]))
    return x, y, z

def execute_shooting(xa, xb, N, lb, ub):
    h = np.divide((xb - xa), (N - 1))
    t = np.linspace(xa, xb, N)
    range_values = np.arange(lb, ub, 0.1)
    X = np.zeros(N)
    X[2] = 2
    Y = np.zeros(N)
    Y[0] = 1
    Z = np.zeros(N)
    f = np.zeros_like(range_values)
    g = np.zeros_like(range_values)

    for index, value in enumerate(range_values):
        Z[0] = value
        X, Y, Z = fun1(X, Y, Z, N, h, t)
        f[index] = (X[N-1] - 2)**2
        g[index] = abs(X[N-1] - 2)
        plt.plot(X, Y, 'g')
        plt.axis([xa, xb, lb, ub])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y(\alpha)$')
        plt.title(r'$y(\alpha)$')
    plt.show()

    fig2, ax1 = plt.subplots()
    ax1.plot(range_values, f, 'b-')
    #ax1.axis([-5, -2, 0, 5])
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$J(\alpha)$')
    ax2 = ax1.twinx()
    ax2.plot(range_values, g, 'r-')
    plt.title(r'$J(\alpha)$')
    plt.show()

execute_shooting(xa=1, xb=3, N=1000, lb=-5, ub=5)
