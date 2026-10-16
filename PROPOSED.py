import numpy as np
import time


def PROPOSED(population, objective_function, lb, ub, max_iter):
    print("Refined Perfumer Optimization Algorithm (RPOA)")

    # Proposed Position update is done at line 45
    population_size, dimension = population.shape
    lb = np.array(lb)
    ub = np.array(ub)

    # Evaluate Initial Population
    fitness = np.zeros(population_size)
    for i in range(population_size):
        fitness[i] = objective_function(population[i])

    best_index = np.argmin(fitness)
    best_solution = population[best_index].copy()
    best_fitness = fitness[best_index]
    worst_index = np.argmax(fitness)
    worst_solution = population[worst_index].copy()
    worst_fitness = fitness[worst_index]
    convergence_curve = np.zeros((max_iter))

    ct = time.time()
    for iteration in range(max_iter):
        # Phase 1 : Exploration

        for i in range(population_size):
            # Reference Formulation (Eq.4)
            r = np.random.rand()
            I = np.random.randint(1, 3)

            RF = population[i] + r * (
                    best_solution - I * worst_solution
            )

            RF = np.maximum(RF, lb[i])
            RF = np.minimum(RF, ub[i])

            # Equation (5)
            # Proposed Position update is done here
            # r = np.random.rand()
            r = np.min(fitness) / (np.max(fitness) + fitness[i])
            I = np.random.randint(1, 3)
            new_solution = population[i].copy()

            for j in range(dimension):
                new_solution[j] = (
                        population[i][j]
                        + I * r *
                        (RF[j] - I * population[i][j])
                )

            # Boundary Check
            new_solution = np.maximum(new_solution, lb[i])
            new_solution = np.minimum(new_solution, ub[i])

            # Evaluate
            new_fitness = objective_function(new_solution)

            # Equation (6)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Update Best/Worst after Exploration
        for i in range(population_size):
            fitness[i] = objective_function(population[i])

        best_index = np.argmin(fitness)
        best_solution = population[best_index].copy()
        best_fitness = fitness[best_index]

        worst_index = np.argmax(fitness)
        worst_solution = population[worst_index].copy()
        worst_fitness = fitness[worst_index]

        # Phase 2 : Exploitation
        t = iteration + 1
        for i in range(population_size):
            new_solution = population[i].copy()
            for j in range(dimension):
                r = np.random.rand()
                I = np.random.randint(1, 3)
                new_solution[j] = (
                        population[i][j]
                        + (1 - (2 * I * r))
                        * ((ub[i, j] - lb[i, j]) / t)
                )

            # Boundary Check
            new_solution = np.maximum(new_solution, lb[i])
            new_solution = np.minimum(new_solution, ub[i])
            new_fitness = objective_function(new_solution)

            # Equation (8)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Update Best/Worst after Exploitation
        for i in range(population_size):
            fitness[i] = objective_function(population[i])

        best_index = np.argmin(fitness)
        best_solution = population[best_index].copy()
        best_fitness = fitness[best_index]
        worst_index = np.argmax(fitness)
        worst_solution = population[worst_index].copy()
        worst_fitness = fitness[worst_index]
        convergence_curve[iteration] = best_fitness

        if (iteration + 1) % 10 == 0:
            print(
                "Iteration {:4d}/{:4d} Best Fitness = {:.10f}".format(
                    iteration + 1,
                    max_iter,
                    best_fitness
                )
            )
    ct = time.time() - ct
    return best_fitness, convergence_curve, best_solution, ct
