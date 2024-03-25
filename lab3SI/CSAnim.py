import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Rastring
def fobj(X):
    n = len(X)
    result = 0
    for xi in X:
        xi = np.array(xi)
        result += xi**2 - 10 * np.cos(2 * np.pi * xi)
    return 10 * n + result
#Mishara Bird
# def fobj(X):
#     x = np.array(X[0])
#     y = np.array(X[1])
#     condition = ( (x + 5)**2 + (y + 5)**2 >= 25 )

#     func_values = np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    
#     func_values = np.array(func_values) 
#     func_values[condition] = 10000
#     return func_values

def get_cuckoos(nest, best, Lb, Ub):
    n = nest.shape[0]
    beta = 3/2
    sigma = (gamma(1+beta)*np.sin(np.pi*beta/2)/(gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)

    new_nest = nest.copy()
    for j in range(n):
        s = nest[j]
        u = np.random.randn(*s.shape) * sigma
        v = np.random.randn(*s.shape)
        step = u / np.abs(v)**(1/beta)
        stepsize = 0.01 * step * (s - best)
        s = s + stepsize * np.random.randn(*s.shape)
        new_nest[j] = simplebounds(s, Lb, Ub)

    return new_nest

def get_best_nest(nest, newnest, fitness):
    n = nest.shape[0]
    fnew = np.apply_along_axis(fobj, 1, newnest)
    mask = fnew <= fitness
    fitness[mask] = fnew[mask]
    nest[mask] = newnest[mask]
    fmin, best = np.min(fitness), nest[np.argmin(fitness)]
    return fmin, best, nest, fitness

def empty_nests(nest, Lb, Ub, pa):
    n = nest.shape[0]
    K = np.random.rand(*nest.shape) > pa
    stepsize = np.random.rand(*nest.shape) * (nest[np.random.permutation(n)] - nest[np.random.permutation(n)])
    new_nest = nest + stepsize * K
    new_nest = np.apply_along_axis(lambda x: simplebounds(x, Lb, Ub), 1, new_nest)
    return new_nest

def simplebounds(s, Lb, Ub):
    s = np.maximum(s, Lb)
    s = np.minimum(s, Ub)
    return s

def cuckoo_search_animation(nPop, MaxIt):
    # Animation Data
    pops = []

    # Initialization
    nest = np.random.uniform(low=Lb, high=Ub, size=(nPop, nd))
    fitness = np.full(nPop, fill_value=np.inf)
    fmin, bestnest, nest, fitness = get_best_nest(nest, nest, fitness)

    for it in range(MaxIt):
        new_nest = get_cuckoos(nest, bestnest, Lb, Ub)
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        new_nest = empty_nests(nest, Lb, Ub, pa)
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)

        pops.append(nest.copy())

        if fnew < fmin:
            fmin = fnew
            bestnest = best

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(-5, 5, 0.01)
    Y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = fobj([X, Y])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)

    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)
        ax.scatter(pops[frame][:, 0], pops[frame][:, 1], fobj(pops[frame].T), marker='*', edgecolors='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title(f'Cuckoo Search Iteration {frame+1}/{MaxIt}')

    ani = animation.FuncAnimation(fig, update, frames=range(MaxIt), interval=200)
    plt.show()

# Algorithm Parameters
pa = 0.25
nd = 2
Lb = np.full(nd, -5)
Ub = np.full(nd, 5)
nPop = 30
MaxIt = 50

# Run Animation
cuckoo_search_animation(nPop, MaxIt)

