import numpy as np

def bat_algorithm_new(inp=None):
    if inp is None:
        inp = [20, 1000]  # Default values for n=10 and t_max=1000

    # Initialize all the default parameters
    n = inp[0]  # Population size, typically 20 to 40
    t_max = inp[1]  # Maximum number of iterations
    A = 1  # Initial loudness (constant or decreasing)
    r0 = 1  # The initial pulse rate (constant or decreasing)
    alpha = 0.97  # Parameter alpha
    gamma = 0.1  # Parameter gamma
    # Frequency range
    Freq_min = 0  # Frequency minimum
    Freq_max = 2  # Frequency maximum
    t = 0  # Initialize iteration counter
    # Dimensions of the search variables
    d = 2

    # Initialization of all the arrays
    Freq = np.zeros(n)  # Frequency-tuning array
    v = np.zeros((n, d))  # Equivalent velocities or increments
    Lb = -5 * np.ones(d)  # Lower bounds
    Ub = 5 * np.ones(d)  # Upper bounds

    # Initialize the population/solutions
    Sol = np.random.rand(n, d) * (Ub - Lb) + Lb
    Fitness = np.apply_along_axis(Fun, 1, Sol)

    # Find the best solution of the initial population
    I = np.argmin(Fitness)
    fmin = Fitness[I]
    best = Sol[I, :]

    # Start the iterations -- the Bat Algorithm (BA) -- main loop
    while t < t_max:
        # Varying loudness (A) and pulse emission rate (r)
        r = r0 * (1 - np.exp(-gamma * t))
        A = alpha * A

        # Loop over all bats/solutions
        for i in range(n):
            Freq[i] = Freq_min + np.random.rand() * (Freq_max - Freq_min)
            v[i, :] += (Sol[i, :] - best) * Freq[i]
            S = Sol[i, :] + v[i, :]

            # Check a switching condition
            if np.random.rand() < r:
                S = best + 0.1 * np.random.randn(d) * A

            # Check if the new solution is within the simple bounds
            S = simplebounds(S, Lb, Ub)

            # Evaluate new solutions
            Fnew = Fun(S)

            # If the solution improves or not too loudness
            if Fnew <= Fitness[i] and np.random.rand() < A:
                Sol[i, :] = S
                Fitness[i] = Fnew

            # Update the current best solution
            if Fnew <= fmin:
                best = S
                fmin = Fnew

        t += 1  # Update iteration counter

        # Display the results every 100 iterations
        if t % 100 == 0:
            print(f"Iteration = {t}")
            print("Best:", best, "fmin:", fmin)

    # Output the best solution
    print("Best =", best, "fmin=", fmin)


# Application of simple bounds/constraints
def simplebounds(s, Lb, Ub):
    # Apply the lower bound
    ns_tmp = np.copy(s)
    ns_tmp = np.where(ns_tmp < Lb, Lb, ns_tmp)

    # Apply the upper bounds
    ns_tmp = np.where(ns_tmp > Ub, Ub, ns_tmp)

    # Update this new move
    return ns_tmp


# The cost function or objective function
def Fun(u):
    # Easom function
    x, y = u
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

# Example usage
bat_algorithm_new()
