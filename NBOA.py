import time
import numpy as np


#  Namib Beetle Optimization Algorithm (NBOA)
def NBOA(population, fobj, VRmin, VRmax, Max_iter):
    population_size, dimension = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    fitness = fobj(population[:])
    best_fitness = float('inf')
    best_solution = np.zeros((dimension, 1))

    evaporation_rate = 0.3
    learning_rate = 0.1

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        new_population = np.copy(population)
        for i in range(population_size):
            candidate = population[i]
            # Randomly select two different beetles
            indices = np.random.choice(population_size, 2, replace=False)
            beetle_a = population[indices[0]]
            beetle_b = population[indices[1]]
            # Update candidate solution based on water collection behavior
            new_candidate = candidate + learning_rate * (beetle_a - beetle_b)
            new_candidate = np.clip(new_candidate, lb, ub)
            new_fitness = fobj(new_candidate)
            if new_fitness < fitness[i]:
                new_population[i] = new_candidate
                fitness[i] = new_fitness
                if new_fitness < best_fitness:
                    best_solution = new_candidate
                    best_fitness = new_fitness
        population = new_population * (1 - evaporation_rate)
        fitness = np.apply_along_axis(fobj, 1, population)

        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct

