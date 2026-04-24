
# %% Homework 1
import numpy as np
import matplotlib.pyplot as plt
import time
def objective_function(x):
    # f(x) = 4/3 * (x1^2 + x2^2 - x1*x2)^0.75 + |x3|
    term1 = (x[0]**2 + x[1]**2 - x[0]*x[1])**0.75
    return (4/3) * term1 + np.abs(x[2])

def simulated_annealing(initial_temp, cooling_rate, max_iterations, step_size=0.1):
    # randomized initial solution within bounds [-1, 1]
    current_x = np.random.uniform(-1, 1, size=3)
    current_f = objective_function(current_x)
    
    best_x = np.copy(current_x)
    best_f = current_f
    
    temp = initial_temp
    
    # Tracking history for visualization
    
    history = []

    while temp > 1e-6:
        for _ in range(max_iterations):
            # Neighbor generation: add Gaussian noise and clip to bounds
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
        
        history.append(best_f)
        temp *= cooling_rate
        
    return best_x, best_f, history

# %% Monte Carlo Simulation
num_runs = 100
results_f = [] 

print(f"Running {num_runs} Monte Carlo simulations...")

for i in range(num_runs):

    best_x, best_f, _ = simulated_annealing(initial_temp=10, cooling_rate=0.95, max_iterations=50)
    
    results_f.append(best_f)

mean_val = np.mean(results_f)
std_val = np.std(results_f)

print("-" * 30)
print(f"Statistics after {num_runs} runs:")
print(f"Mean of the minimum values: {mean_val:.6f}")
print(f"Standard deviation: {std_val:.6f}")
print(f"Best result ever achieved: {np.min(results_f):.6f}")

# %% Visualization of SA results and Monte Carlo distribution

# 1. Convergence of SA (alpha=0.95, N=50 iterations)
best_x, best_f, history = simulated_annealing(initial_temp=10, cooling_rate=0.95, max_iterations=50)

plt.figure(figsize=(10, 5))
plt.plot(history, color='blue', lw=1.5)
plt.title('Convergence of Simulated Annealing (alpha=0.95)')
plt.xlabel('Temperature step (Cooling Iteration)')
plt.ylabel('Current value f(x)')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Comparison of cooling rates (alpha) in SA convergence 
# Testing: Superfast (0.5), Good (0.95) and Slow (0.999)
rates = [0.5, 0.95, 0.999]
plt.figure(figsize=(12, 6))

for r in rates:
    _, _, h = simulated_annealing(initial_temp=10, cooling_rate=r, max_iterations=20)
    plt.plot(h, label=f'Cooling rate (alpha) = {r}')

plt.yscale('log') 
plt.title('Comparison of Cooling Rates in Simulated Annealing')
plt.xlabel('Iteration')
plt.ylabel('f(x) value (Log scale)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

# 3. Histogram Monte Carlo results 
plt.figure(figsize=(10, 5))
plt.hist(results_f, bins=20, color='green', edgecolor='black', alpha=0.7)
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
plt.title(f'Distribution of minimum over {num_runs} Monte Carlo simulations')
plt.xlabel('Found minimum value f(x)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# %% Genetski algoritam

def genetic_algorithm(population_size, generations, mutation_rate, tournament_size=3):
    # Inicijalization
    population = np.random.uniform(-1, 1, (population_size, 3))
    
    # Elitism - saving the best solution found so far
    best_ind = None
    best_fit = float('inf')
    
    
    avg_fitness_history = []

    for gen in range(generations):
        fitness = np.array([objective_function(ind) for ind in population])
        
        # Update best (Elitism)
        current_min_idx = np.argmin(fitness)
        if fitness[current_min_idx] < best_fit:
            best_fit = fitness[current_min_idx]
            best_ind = np.copy(population[current_min_idx])
            
        avg_fitness_history.append(np.mean(fitness))

        # 1. SELECTION (Tournament)
        new_population = []
        for _ in range(population_size):
            # Choose random candidates for the tournament
            candidates_idx = np.random.choice(population_size, tournament_size)
            # Winner is the one with the best fitness (lowest f(x))
            winner_idx = candidates_idx[np.argmin(fitness[candidates_idx])]
            new_population.append(population[winner_idx])
        
        new_population = np.array(new_population)

        # 2. CROSSOVER (Arithmetic)
        offspring = []
        for i in range(0, population_size, 2):
            p1 = new_population[i]
            # Handle even/odd population size by wrapping around
            p2 = new_population[i+1] if (i+1) < population_size else new_population[0]
            
            alpha = np.random.rand()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = alpha * p2 + (1 - alpha) * p1
            offspring.append(child1)
            offspring.append(child2)
            
        # Shorten offspring to population size (in case of odd population)
        offspring = np.array(offspring[:population_size])

        # 3. MUTATION
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                # Dodajemo Gausov šum
                offspring[i] += np.random.normal(0, 0.1, size=3)
                offspring[i] = np.clip(offspring[i], -1, 1)

        # Insert elitism
        population = offspring
        population[np.random.randint(population_size)] = best_ind

    return best_ind, best_fit, avg_fitness_history

# %% Monte Carlo simulation for GA


def run_mc_ga():
    # Parametres for GA Monte Carlo simulation
    pop_sizes = [10, 20, 50, 100]
    gens = 100
    num_mc = 20
    
    results = []
    complexities = []

    print("Running MC for GA...")
    for pop in pop_sizes:
        current_pop_results = []
        for _ in range(num_mc):
            _, b_fit, _ = genetic_algorithm(pop, gens, 0.1)
            current_pop_results.append(b_fit)
        
        results.append(np.mean(current_pop_results))
        complexities.append(pop * gens) # Broj računskih operacija

    # GRAPH 1: Efficiency of GA (Operations vs. Quality of Solution)
    plt.figure(figsize=(10, 5))
    plt.plot(complexities, results, 'o-', color='red', label='Average f(x) after GA')
    plt.xlabel('Number of Operations (Population * Generations)')
    plt.ylabel('Average value of f(x) (min f(x))')
    plt.title('Efficiency of GA: Operations vs. Quality of Solution')
    plt.grid(True)
    plt.legend()
    plt.show()

    # GRAPH 2: Impact of mutation rate on convergence (N=20 MC runs, pop_size=50, gens=100)
    mutations = [0.01, 0.1, 0.5]
    plt.figure(figsize=(10, 5))
    for m in mutations:
        _, _, history = genetic_algorithm(50, 100, m)
        plt.plot(history, label=f'Mutation rate= {m}')
    
    plt.xlabel('Generations')
    plt.ylabel('Average value of f(x) (min f(x))')
    plt.title('Impact of Mutation Rate on GA Convergence')
    plt.legend()
    plt.show()

run_mc_ga()

# %% Particle Swarm Optimization (PSO) 
# Vektorized PSO implementation for efficiency and clarity

def pso(num_particles=30, iterations=100, n_dims=3, w=0.7, v_max=0.2):
    particles  = np.random.uniform(-1, 1, (num_particles, n_dims))
    velocities = np.random.uniform(-0.1, 0.1, (num_particles, n_dims))

    pbest_pos     = np.copy(particles)
    pbest_fitness = np.array([objective_function(p) for p in particles])
    gbest_idx     = np.argmin(pbest_fitness)
    gbest_pos     = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    history = [gbest_fitness]
    for _ in range(iterations):
        r1 = np.random.rand(num_particles, n_dims)
        r2 = np.random.rand(num_particles, n_dims)

        # Update velocities and positions
        velocities = (w * velocities
                      + 2 * r1 * (pbest_pos - particles)
                      + 2 * r2 * (gbest_pos - particles))
        velocities = np.clip(velocities, -v_max, v_max)  # velocity clamping
        particles += velocities
        particles  = np.clip(particles, -1, 1)

        # Update pbest and gbest
        fitness = np.array([objective_function(p) for p in particles])
        improved = fitness < pbest_fitness
        pbest_pos[improved]     = np.copy(particles[improved])
        pbest_fitness[improved] = fitness[improved]

        best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[best_idx]
            gbest_pos     = np.copy(pbest_pos[best_idx])
        history.append(gbest_fitness)
    return gbest_pos, gbest_fitness, history
# %% Visualization of PSO results and Monte Carlo distribution

print("=== Sanity check ===")
print(f"f([0,0,0])   = {objective_function([0,0,0]):.6f}  (expected: 0.000000)")
print(f"f([1,1,1])   = {objective_function([1,1,1]):.6f}  (expected: 2.333333)")
print(f"f([-1,-1,0]) = {objective_function([-1,-1,0]):.6f}  (expected: 1.333333)")
print()
 
 
# Monte Carlo simulation of PSO algorithm (N=50 iterations)
N_MC = 50
results = [pso(num_particles=30, iterations=100) for _ in range(N_MC)]
fitnesses = [r[1] for r in results]
best_run  = results[np.argmin(fitnesses)]
 
print("=== Monte Carlo results (N=50, 30 particles, 100 iterations) ===")
print(f"Average f values: {np.mean(fitnesses):.6f}")
print(f"Stdandard deviation of f values     : {np.std(fitnesses):.6f}")
print(f"Minimum f               : {np.min(fitnesses):.6f}")
print(f"Maximum f               : {np.max(fitnesses):.6f}")
print(f"Best f value  : {np.round(best_run[0], 5)}")
print()
 
# GRAPH 1 - Convergence of PSO (N=50 Monte Carlo runs)
histories = np.array([r[2] for r in results])
mean_hist = histories.mean(axis=0)
std_hist  = histories.std(axis=0)
iters     = np.arange(len(mean_hist))
 
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(iters, mean_hist, color="#1f77b4", linewidth=2, label="Average values of f(gbest)")
ax.fill_between(iters,
                mean_hist - std_hist,
                mean_hist + std_hist,
                alpha=0.2, color="#1f77b4", label="±1 std")
ax.set_xlabel("Iteration")
ax.set_ylabel("f(gbest)")
ax.set_title("Convergence of PSO (N=50 Monte Carlo simulations)")
ax.legend()
plt.tight_layout()
plt.show()
 
 
# GRAPH 2 — Impact of number of particles on PSO performance (N=30 Monte Carlo runs, 100 iterations)
particle_counts = [5, 10, 20, 30, 50, 100]
N_MC2 = 30
 
mean_fits, std_fits = [], []
for np_ in particle_counts:
    fs = [pso(num_particles=np_, iterations=100)[1] for _ in range(N_MC2)]
    mean_fits.append(np.mean(fs))
    std_fits.append(np.std(fs))
 
mean_fits = np.array(mean_fits)
std_fits  = np.array(std_fits)
 
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(particle_counts, mean_fits, yerr=std_fits,
            fmt="o-", color="#2ca02c", capsize=5, linewidth=2, markersize=6)
ax.set_xlabel("Number of particles(N)")
ax.set_ylabel("Average value f(x)")
ax.set_title("Impact of number of particles on PSO performance(N=30 MC, 100 iteracija)")
plt.tight_layout()
plt.show()
            

# %% Comparison of SA, GA and PSO (N=50 Monte Carlo runs each)

def compare_sa_ga_pso(num_runs=50):
    sa_results = []
    ga_results = []
    pso_results = []
    

    sa_histories = []
    ga_histories = []
    pso_histories = []

    print(f"Pokrećem poređenje: {num_runs} MC simulacija po algoritmu...")

    # TESTING SA
    start_sa = time.time()
    for i in range(num_runs):
        _, best_f_sa, history = simulated_annealing(initial_temp=10, cooling_rate=0.95, max_iterations=50)
        sa_results.append(best_f_sa)
        if i < 5: sa_histories.append(history) # Saving first 5 for graph
    end_sa = time.time()

    # TESTING GA
    # Aligning GA evaluations with SA:

    # SA: ~315 cooling steps * 50 iterations = 15,750 evals
    # GA: pop_size=50 implies 15,750 / 50 = 315 generations
    num_generations_ga = len(sa_histories[0]) 
    
    start_ga = time.time()
    for i in range(num_runs):
        _, best_f_ga, history = genetic_algorithm(population_size=50, generations=num_generations_ga, mutation_rate=0.1)
        ga_results.append(best_f_ga)
        if i < 5: ga_histories.append(history) # Saving first 5 for graph
    end_ga = time.time()

    # TESTING PSO
    # 15750 / 30 (number of particles) = 525 iterations
    start_pso = time.time()
    for i in range(num_runs):
        _, best_f_pso, history = pso(num_particles=30, iterations=525)
        pso_results.append(best_f_pso)
        if i < 5: pso_histories.append(history) # Saving first 5 for graph
    end_pso = time.time()
    

    # VISUALIZATION: Boxplot + Convergence Graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Grafh 1: Boxplot 
    ax1.boxplot([sa_results, ga_results, pso_results], labels=['SA', 'GA', 'PSO'])
    ax1.set_ylabel('Pronađeni minimum f(x)')
    ax1.set_title('Preciznost i stabilnost (Boxplot)')
    ax1.grid(True, axis='y', alpha=0.3)

    # Graph 2: Convergence
    for i in range(5):
        ax2.plot(sa_histories[i], color='blue', alpha=0.3, label='SA' if i == 0 else "")
        ax2.plot(ga_histories[i], color='red', alpha=0.3, label='GA' if i == 0 else "")
        ax2.plot(pso_histories[i], color='green', alpha=0.3, label='PSO' if i == 0 else "")
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (Step of cooling / Generations / Iterations)')
    ax2.set_ylabel('Values f(x) (Log scale)')
    ax2.set_title('Convergence of SA, GA and PSO (First 5 runs)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.1)

    plt.tight_layout()
    plt.show()

    
    evals_per_run = len(sa_histories[0]) * 50
    print("-" * 40)
    print(f"Analysis of {num_runs} Monte Carlo runs:")
    print(f"Number of evaluations per run: {evals_per_run}")
    print(f"Total number of evaluations in MC test: {evals_per_run * num_runs}")
    print("-" * 40)
    print(f"Results (N={num_runs}):")
    print(f"SA Average value: {np.mean(sa_results):.4f} (Std: {np.std(sa_results):.4f})")
    print(f"GA Average value: {np.mean(ga_results):.4f} (Std: {np.std(ga_results):.4f})")
    print(f"PSO Average value: {np.mean(pso_results):.4f} (Std: {np.std(pso_results):.4f})")
    print(f"Time taken: SA={end_sa-start_sa:.2f}s, GA={end_ga-start_ga:.2f}s, PSO={end_pso-start_pso:.2f}s")
    print("-" * 40)

compare_sa_ga_pso()
