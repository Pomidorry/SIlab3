import numpy as np
from scipy.special import gamma

def fobj(u):
    # Easom function
    x, y = u
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

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

def cuckoo_search_new(inp=None):
    if inp is None:
        n = 25  # Default values for n
        N_IterTotal = 1000  # Default value for N_IterTotal
    else:
        n = inp[0]
        N_IterTotal = inp[1]

    pa = 0.25  # Discovery rate of alien eggs/solutions
    nd = 2  # Dimensions of the problem
    Lb = np.full(nd, -5)  # Lower bounds
    Ub = np.full(nd, 5)  # Upper bounds

    # Random initial solutions
    nest = np.random.uniform(low=Lb, high=Ub, size=(n, nd))
    # Get the current best of the initial population
    fitness = np.full(n, fill_value=np.inf)
    fmin, bestnest, nest, fitness = get_best_nest(nest, nest, fitness)

    # Starting iterations
    for iter in range(1, N_IterTotal + 1):
        # Generate new solutions (but keep the current best)
        new_nest = get_cuckoos(nest, bestnest, Lb, Ub)
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        # Discovery and randomization
        new_nest = empty_nests(nest, Lb, Ub, pa)
        # Evaluate this set of solutions
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        # Find the best objective so far
        if fnew < fmin:
            fmin = fnew
            bestnest = best
        # Display the results every 100 iterations
        if iter % 100 == 0:
            print("Iteration =", iter)
            print("Minimum value found:", fmin)

    # Post-optimization processing and display all the nests
    print("The best solution:", bestnest)
    print("The best fmin:", fmin)

# Example usage
if __name__ == "__main__":
    cuckoo_search_new([25, 1000])
