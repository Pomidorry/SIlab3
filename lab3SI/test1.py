import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def bat_algorithm_animation(nPop, MaxIt, d):
    # Bat Algorithm Parameters
    A = 1
    r0 = 1
    alpha = 0.97
    gamma = 0.1
    Freq_min = 0
    Freq_max = 2
    Lb = -5.12 * np.ones(d)
    Ub = 5.12 * np.ones(d)
    t = 0

    # Animation Data
    pops = []

    # Initialize Population and Fitness
    Sol = np.random.rand(nPop, d) * (Ub - Lb) + Lb
    Fitness = np.apply_along_axis(Fun, 1, Sol)

    # Find the best solution of the initial population
    I = np.argmin(Fitness)
    fmin = Fitness[I]
    best = Sol[I, :]

    for t in range(MaxIt):
        r = r0 * (1 - np.exp(-gamma * t))
        A = alpha * A

        for i in range(nPop):
            v = np.zeros(d)
            Freq = Freq_min + np.random.rand() * (Freq_max - Freq_min)
            v = v + (Sol[i, :] - best) * Freq
            S = Sol[i, :] + v

            if np.random.rand() < r:
                S = best + 0.1 * np.random.randn(d) * A

            S = simplebounds(S, Lb, Ub)
            Fnew = Fun(S)

            if Fnew <= Fitness[i] and np.random.rand() < A:
                Sol[i, :] = S
                Fitness[i] = Fnew

            if Fnew <= fmin:
                best = S
                fmin = Fnew

        pops.append(Sol.copy())

        # if t % 2 == 0:
        #     print(f"Iteration = {t}")
        #     print("Best:", best, "fmin:", fmin)

    if d == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = np.arange(-5, 5, 0.01)
        Y = np.arange(-5, 5, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = Fun([X, Y])
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)

        def update(frame):
            ax.clear()
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)
            ax.scatter(pops[frame][:, 0], pops[frame][:, 1], Fun(pops[frame].T), marker='*', edgecolors='red')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Fitness')
            ax.set_title(f'Bat Algorithm Iteration {frame+1}/{MaxIt}')

        ani = animation.FuncAnimation(fig, update, frames=range(MaxIt), interval=200)
        plt.show()
    else:
        # Plot fitness trend
        fmins = []
        for it in range(MaxIt):
            for i in range(nPop):
                v = np.zeros(d)
                Freq = Freq_min + np.random.rand() * (Freq_max - Freq_min)
                v = v + (Sol[i, :] - best) * Freq
                S = Sol[i, :] + v

                if np.random.rand() < r:
                    S = best + 0.1 * np.random.randn(d) * A

                S = simplebounds(S, Lb, Ub)
                Fnew = Fun(S)

                if Fnew <= Fitness[i] and np.random.rand() < A:
                    Sol[i, :] = S
                    Fitness[i] = Fnew

                if Fnew <= fmin:
                    best = S
                    fmin = Fnew

            fmins.append(fmin)

        plt.plot(range(MaxIt), fmins)
        plt.xlabel('Покоління')
        plt.ylabel('Min пристосованість')
        plt.title('Залежність min')
        plt.show()

# Application of simple bounds/constraints
def simplebounds(s, Lb, Ub):
    ns_tmp = np.copy(s)
    ns_tmp = np.where(ns_tmp < Lb, Lb, ns_tmp)
    ns_tmp = np.where(ns_tmp > Ub, Ub, ns_tmp)
    return ns_tmp

# Rastring
def Fun(X):
    n = len(X)
    result = 0
    for xi in X:
        xi = np.array(xi)
        result += xi**2 - 10 * np.cos(2 * np.pi * xi)
    return 10 * n + result
# Mishara Bird
# def Fun(X):
#     x = np.array(X[0])
#     y = np.array(X[1])
#     condition = ( (x + 5)**2 + (y + 5)**2 >= 25 )

#     func_values = np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    
#     func_values = np.array(func_values) 
#     func_values[condition] = 10000
#     return func_values

# Example usage
bat_algorithm_animation(1000, 150, 2)
