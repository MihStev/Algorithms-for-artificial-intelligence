
# %% Homework 1: Simulated Annealing for Function Optimization
import numpy as np
import matplotlib.pyplot as plt
import time
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

# %% Genetic Algorithm Implementation

def genetic_algorithm(population_size, generations, mutation_rate, tournament_size=3):
    # Inicijalizacija
    population = np.random.uniform(-1, 1, (population_size, 3))
    
    # Elitism: tracking the best ever found solution
    best_ind = None
    best_fit = float('inf')
    
    # For the plot: tracking average fitness across generations
    avg_fitness_history = []

    for gen in range(generations):
        fitness = np.array([objective_function(ind) for ind in population])
        
        # Update najboljeg (Elitizam)
        current_min_idx = np.argmin(fitness)
        if fitness[current_min_idx] < best_fit:
            best_fit = fitness[current_min_idx]
            best_ind = np.copy(population[current_min_idx])
            
        avg_fitness_history.append(np.mean(fitness))

        # 1. SELEKCIJA (Tournament)
        new_population = []
        for _ in range(population_size):
            # Izaberemo nasumičnih k jedinki
            candidates_idx = np.random.choice(population_size, tournament_size)
            # Pobednik je onaj sa najmanjim fitnessom
            winner_idx = candidates_idx[np.argmin(fitness[candidates_idx])]
            new_population.append(population[winner_idx])
        
        new_population = np.array(new_population)

        # 2. UKRŠTANJE (Aritmetičko - bolje za realne brojeve)
        offspring = []
        for i in range(0, population_size, 2):
            p1 = new_population[i]
            # Handle neparan broj populacije
            p2 = new_population[i+1] if (i+1) < population_size else new_population[0]
            
            alpha = np.random.rand()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = alpha * p2 + (1 - alpha) * p1
            offspring.append(child1)
            offspring.append(child2)
            
        # Skrati ako je dodat jedan višak zbog neparnog broja
        offspring = np.array(offspring[:population_size])

        # 3. MUTACIJA
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                # Dodajemo Gausov šum
                offspring[i] += np.random.normal(0, 0.1, size=3)
                offspring[i] = np.clip(offspring[i], -1, 1)

        # Ubaci najboljeg iz prethodne generacije (Elitizam) nazad na slučajno mesto
        population = offspring
        population[np.random.randint(population_size)] = best_ind

    return best_ind, best_fit, avg_fitness_history

# %% Monte Carlo za Genetski Algoritam (Traženi grafici)


def run_mc_ga():
    # Parametri za testiranje efikasnosti
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

    # GRAFIK 1: Računarska efikasnost (Traženo u tekstu zadatka)
    plt.figure(figsize=(10, 5))
    plt.plot(complexities, results, 'o-', color='red', label='Average Best Fitness')
    plt.xlabel('Number of Computational Operations (Pop_size * Gen)')
    plt.ylabel('Average Fitness (min f(x))')
    plt.title('Efficiency of the Genetic Algorithm')
    plt.grid(True)
    plt.legend()
    plt.show()

    # GRAFIK 2: Uticaj verovatnoće mutacije
    mutations = [0.01, 0.1, 0.5]
    plt.figure(figsize=(10, 5))
    for m in mutations:
        _, _, history = genetic_algorithm(50, 100, m)
        plt.plot(history, label=f'Mutation Rate = {m}')
    
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness of Population')
    plt.title('Impact of Mutation Rate on Convergence')
    plt.legend()
    plt.show()

run_mc_ga()

# %% Comparison of Simulated Annealing and Genetic Algorithm
# %% [markdown]
# Radi fer poređenja algoritama na statistički značajan način, parametri su podešeni tako da oba algoritma vrše približno isti broj evaluacija funkcije cilja (
# N\approx15.750
# ). Za Simulirano kaljenje, pri hlađenju sa T=10 na T={10}^{-6} uz faktor \alpha=0.95,
# algoritam izvršava oko 315 koraka hlađenja sa po 50 unutrašnjih iteracija. 
# Da bi se postigla ista kompleksnost, Genetski algoritam je konfigurisan sa populacijom od 50 jedinki
# kroz 315 generacija. Na ovaj način, razlika u kvalitetu rešenja potiče isključivo od efikasnosti
# same strategije pretrage, a ne od količine uloženih računarskih resursa.

#%%
def compare_sa_ga(num_runs=50):
    sa_results = []
    ga_results = []
    
    # Za intuitivni grafik konvergencije uzećemo po 5 reprezentativnih pokretanja
    sa_histories = []
    ga_histories = []

    print(f"Pokrećem poređenje: {num_runs} MC simulacija po algoritmu...")

    # --- TESTIRANJE SA ---
    start_sa = time.time()
    for i in range(num_runs):
        _, best_f_sa, history = simulated_annealing(initial_temp=10, cooling_rate=0.95, max_iterations=50)
        sa_results.append(best_f_sa)
        if i < 5: sa_histories.append(history) # Čuvamo prvih 5 za grafik
    end_sa = time.time()

    # --- TESTIRANJE GA ---
    # Izračunavamo broj generacija tako da GA ima isti broj evaluacija kao SA
    # SA ima ~315 koraka hlađenja * 50 iteracija = 15750 evaluacija
    # GA sa pop_size=50 treba da ima 15750 / 50 = 315 generacija
    num_generations_ga = len(sa_histories[0]) 
    
    start_ga = time.time()
    for i in range(num_runs):
        _, best_f_ga, history = genetic_algorithm(population_size=50, generations=num_generations_ga, mutation_rate=0.1)
        ga_results.append(best_f_ga)
        if i < 5: ga_histories.append(history) # Čuvamo prvih 5 za grafik
    end_ga = time.time()

    # --- VIZUELIZACIJA (Dva grafika jedan pored drugog) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Grafik 1: Boxplot (Statistička značajnost)
    ax1.boxplot([sa_results, ga_results], labels=['SA', 'GA'])
    ax1.set_ylabel('Pronađeni minimum f(x)')
    ax1.set_title('Preciznost i stabilnost (Boxplot)')
    ax1.grid(True, axis='y', alpha=0.3)

    # Grafik 2: Konvergencija (Intuitivno poređenje)
    for i in range(5):
        ax2.plot(sa_histories[i], color='blue', alpha=0.3, label='SA' if i == 0 else "")
        ax2.plot(ga_histories[i], color='red', alpha=0.3, label='GA' if i == 0 else "")
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Vreme (Korak hlađenja / Generacija)')
    ax2.set_ylabel('Vrednost f(x) (Log skala)')
    ax2.set_title('Borba algoritama kroz vreme')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.1)

    plt.tight_layout()
    plt.show()

    # --- STATISTIKA ZA IZVEŠTAJ ---
    evals_per_run = len(sa_histories[0]) * 50
    print("-" * 40)
    print(f"ANALIZA KOMPLEKSNOSTI:")
    print(f"Broj evaluacija po pokretanju: {evals_per_run}")
    print(f"Ukupno evaluacija u MC testu: {evals_per_run * num_runs}")
    print("-" * 40)
    print(f"REZULTATI (N={num_runs}):")
    print(f"SA Average: {np.mean(sa_results):.8f} (Std: {np.std(sa_results):.8f})")
    print(f"GA Average: {np.mean(ga_results):.8f} (Std: {np.std(ga_results):.8f})")
    print(f"Vreme: SA={end_sa-start_sa:.2f}s, GA={end_ga-start_ga:.2f}s")
    print("-" * 40)

compare_sa_ga()

# %% Particle Swarm Optimization (PSO) Placeholder
# Vectorized implementation of PSO for optimization
# instead of for loops

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
# %% Visualization of PSO convergence
# Sanity check
print("=== Sanity check ===")
print(f"f([0,0,0])   = {objective_function([0,0,0]):.6f}  (expected: 0.000000)")
print(f"f([1,1,1])   = {objective_function([1,1,1]):.6f}  (expected: 2.333333)")
print(f"f([-1,-1,0]) = {objective_function([-1,-1,0]):.6f}  (expected: 1.333333)")
print()
 
 
# Monte Carlo simulation of PSO (N=50 iterations)
N_MC = 50
results = [pso(num_particles=30, iterations=100) for _ in range(N_MC)]
fitnesses = [r[1] for r in results]
best_run  = results[np.argmin(fitnesses)]
 
print("=== Monte Carlo results (N=50, 30 particles, 100 iterations) ===")
print(f"Average f value : {np.mean(fitnesses):.6f}")
print(f"Standard deviation      : {np.std(fitnesses):.6f}")
print(f"Minimum f               : {np.min(fitnesses):.6f}")
print(f"Maximum f               : {np.max(fitnesses):.6f}")
print(f"Best position   : {np.round(best_run[0], 5)}")
print()
 
# GRAPH 1 - Convergence of PSO (N=50 Monte Carlo runs)
histories = np.array([r[2] for r in results])
mean_hist = histories.mean(axis=0)
std_hist  = histories.std(axis=0)
iters     = np.arange(len(mean_hist))
 
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(iters, mean_hist, color="#1f77b4", linewidth=2, label="Average gbest")
ax.fill_between(iters,
                mean_hist - std_hist,
                mean_hist + std_hist,
                alpha=0.2, color="#1f77b4", label="±1 std")
ax.set_xlabel("Iteration")
ax.set_ylabel("f(gbest)")
ax.set_title("Convergence of PSO (N=50 Monte Carlo runs)")
ax.legend()
plt.tight_layout()
plt.show()
 
 
# GRAPH 2 — Impact of Particle Count
particle_counts = [5, 10, 20, 30, 50, 100]
N_MC2 = 30
 
mean_fits, std_fits = [], []
for np_ in particle_counts:
    fs = [pso(num_particles=np_, iterations=100)[1] for _ in range(N_MC2)]
    mean_fits.append(np.mean(fs))
    std_fits.append(np.std(fs))
 
mean_fits = np.array(mean_fits)
std_fits  = np.array(std_fits)
 
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(particle_counts, mean_fits, yerr=std_fits,
            fmt="o-", color="#2ca02c", capsize=5, linewidth=2, markersize=6)
ax.set_xlabel("Number of Particles (N)")
ax.set_ylabel("Average f value")
ax.set_title("Impact of Particle Count (N=30 MC, 100 iterations)")
plt.tight_layout()
plt.show()
            