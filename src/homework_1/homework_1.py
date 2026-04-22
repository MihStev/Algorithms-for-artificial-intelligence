
# %% Homework 1: Simulated Annealing for Function Optimization
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    # f(x) = 4/3 * (x1^2 + x2^2 - x1*x2)^0.75 + |x3|
    term1 = (x[0]**2 + x[1]**2 - x[0]*x[1])**0.75
    return (4/3) * term1 + np.abs(x[2])

def simulated_annealing(initial_temp, cooling_rate, max_iterations, step_size=0.1):
    # Start from a random point in [-1, 1] as per instructions
    current_x = np.random.uniform(-1, 1, size=3)
    current_f = objective_function(current_x)
    
    best_x = np.copy(current_x)
    best_f = current_f
    
    temp = initial_temp
    
    # We track the history for the report plots
    
    history = []

    while temp > 1e-6:
        for _ in range(max_iterations):
            # Neighbor generation: Random walk within bounds
            new_x = current_x + np.random.normal(0, step_size, size=3)
            new_x = np.clip(new_x, -1, 1) 
            
            new_f = objective_function(new_x)
            delta = new_f - current_f
            
            if delta <= 0 or np.random.rand() < np.exp(-delta / temp):
                current_x = new_x
                current_f = new_f
                
                if current_f < best_f:
                    best_f = current_f
                    best_x = np.copy(current_x)
        
        history.append(current_f)
        temp *= cooling_rate
        
    return best_x, best_f, history

# %% Monte Carlo Simulation
num_runs = 100
results_f = [] # List for storing f values

print(f"Running {num_runs} Monte Carlo simulations...")

for i in range(num_runs):

    best_x, best_f, _ = simulated_annealing(initial_temp=100, cooling_rate=0.95, max_iterations=50)
    
    results_f.append(best_f)

mean_val = np.mean(results_f)
std_val = np.std(results_f)

print("-" * 30)
print(f"Statistics after {num_runs} runs:")
print(f"Mean of the minimum values: {mean_val:.6f}")
print(f"Standard deviation: {std_val:.6f}")
print(f"Best result ever achieved: {np.min(results_f):.6f}")
# %%
# %% Visualization and Analysis

# 1. Graphic representation of the convergence (Dokaz za izveštaj)
best_x, best_f, history = simulated_annealing(initial_temp=10, cooling_rate=0.95, max_iterations=50)

plt.figure(figsize=(10, 5))
plt.plot(history, color='blue', lw=1.5)
plt.title('Convergence of Simulated Annealing (alpha=0.95)')
plt.xlabel('Temperature Step (Cooling Iteration)')
plt.ylabel('Current f(x)')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Comaprison of cooling schedules (Dokaz za izveštaj)
# Testing: Superfast (0.5), Good (0.95) and Slow (0.999)
rates = [0.5, 0.95, 0.999]
plt.figure(figsize=(12, 6))

for r in rates:
    _, _, h = simulated_annealing(initial_temp=10, cooling_rate=r, max_iterations=20)
    plt.plot(h, label=f'Cooling rate (alpha) = {r}')

plt.yscale('log') 
plt.title('Comparison of Cooling Schedules')
plt.xlabel('Iteration')
plt.ylabel('f(x) value (Log scale)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

# 3. Histogram Monte Carlo results (Dokaz za izveštaj)
plt.figure(figsize=(10, 5))
plt.hist(results_f, bins=20, color='green', edgecolor='black', alpha=0.7)
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
plt.title(f'Distribution of Minimums over {num_runs} Monte Carlo Runs')
plt.xlabel('Found Minimum f(x)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# %% 

def genetic_alghorithm(population_size, generations, muation_rate):
    # Placeholder for genetic algorithm implementation
    # initialize population, evaluate fitness, selection, crossover, mutation
    population = np.random.uniform(-1, 1, (population_size, 3))
    best_individual = None
    best_fitness = float('inf')

    fitness = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]
    for gen in range(generations):
        # Selection (tournament selection)
        selected_indices = np.random.choice(population_size, size=population_size, replace=True, p=(1/fitness)/np.sum(1/fitness))
        selected_population = population[selected_indices]

        # Crossover (single point)
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1]
            crossover_point = np.random.randint(1, 3) 
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offspring.append(child1)
            offspring.append(child2)

        # Mutation
        for i in range(population_size):
            if np.random.rand() < muation_rate:
                mutation_vector = np.random.normal(0, 0.1, size=3)
                offspring[i] += mutation_vector
                offspring[i] = np.clip(offspring[i], -1, 1)

        population = np.array(offspring)
        fitness = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_individual = population[best_idx]
    return best_individual, best_fitness

# %% Testing the genetic algorithm

best_individual, best_fitness = genetic_alghorithm(population_size=50, generations=100, muation_rate=0.1)
print(f"Best individual found: {best_individual}")
print(f"Best fitness (minimum f(x)): {best_fitness:.6f}")
