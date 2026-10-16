import numpy as np
import time


def RBMO(X, fobj, Xmin, Xmax, T):
    # Red-Billed Magpie Optimization (RBMO) Algorithm.

    # Initialize parameters
    N, D = X.shape
    Xfood = np.zeros(D)
    BestValue = np.inf
    Conv = np.zeros(T)
    fitness = np.full((N), np.inf)
    FES = 0
    Epsilon = 0.5

    # Initialize positions of the agents
    # X = initialization(N, D, Xmin, Xmax)

    X_old = X.copy()
    fitness_old = np.array([fobj(X_old[i, :]) for i in range(N)])
    ct = time.time()
    t = 0  # Start of the main loop
    while t < T:
        # Search for food
        for i in range(N):
            # Randomly select 2 to 5 agents
            p = np.random.randint(2, 6)
            selected_index_p = np.random.choice(N, p, replace=False)
            Xp = X[selected_index_p, :]
            Xpmean = np.mean(Xp, axis=0)

            # Randomly select 10 to N agents
            q = np.random.randint(10, N + 1)
            selected_index_q = np.random.choice(N, q, replace=False)
            Xq = X[selected_index_q, :]
            Xqmean = np.mean(Xq, axis=0)

            R1 = np.random.randint(0, N)
            if np.random.rand() < Epsilon:
                X[i, :] = X[i, :] + (Xpmean - X[R1, :]) * np.random.rand()
            else:
                X[i, :] = X[i, :] + (Xqmean - X[R1, :]) * np.random.rand()

        # Boundary handling
        X = boundary_check(X, Xmin, Xmax)

        # Evaluate fitness
        for i in range(N):
            fitness[i] = fobj(X[i, :])
            if fitness[i] < BestValue:
                BestValue = fitness[i]
                Xfood = X[i, :].copy()
            FES += 1

        # Food storage
        fitness, X, fitness_old, X_old = food_storage(fitness, X, fitness_old, X_old)

        # Calculate control factor
        CF = (1 - t / T) ** (2 * t / T)

        # Exploitation
        for i in range(N):
            # Randomly select 2 to 5 agents
            p = np.random.randint(2, 6)
            selected_index_p = np.random.choice(N, p, replace=False)
            Xp = X[selected_index_p, :]
            Xpmean = np.mean(Xp, axis=0)

            # Randomly select 10 to N agents
            q = np.random.randint(10, N + 1)
            selected_index_q = np.random.choice(N, q, replace=False)
            Xq = X[selected_index_q, :]
            Xqmean = np.mean(Xq, axis=0)

            if np.random.rand() < Epsilon:
                X[i, :] = Xfood + CF * (Xpmean - X[i, :]) * np.random.randn(D)
            else:
                X[i, :] = Xfood + CF * (Xqmean - X[i, :]) * np.random.randn(D)

        # Boundary handling
        X = boundary_check(X, Xmin, Xmax)

        # Evaluate fitness
        for i in range(N):
            fitness[i] = fobj(X[i, :])
            if fitness[i] < BestValue:
                BestValue = fitness[i]
                Xfood = X[i, :].copy()
            FES += 1

        # Food storage
        fitness, X, fitness_old, X_old = food_storage(fitness, X, fitness_old, X_old)

        # Store best fitness
        Conv[t] = BestValue
        t += 1
    ct = time.time() - ct
    return BestValue, Conv, Xfood, ct


def boundary_check(X, Xmin, Xmax):
    """Ensure solutions remain within bounds."""
    return np.clip(X, Xmin, Xmax)


def food_storage(fitness, X, fitness_old, X_old):
    """Implement food storage mechanism to update positions and fitness."""
    improved = fitness < fitness_old
    X_old[improved[:]] = X[improved[:]]
    fitness_old[improved] = fitness[improved]
    return fitness, X, fitness_old, X_old
