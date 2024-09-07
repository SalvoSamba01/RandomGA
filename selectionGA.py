import random
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
random.seed(datetime.now().timestamp())
import time

k = 1000  # Numero di generazioni
n = 350   # Numero di individui
v = 40   # Dimensione del buffer
epsilon = 0.2 # Soglia del parametro
p_fc = 0.7  # Fattore di probabilità di crossover
p_lc = 20  # Livello di probabilità di crossover
p_fm = 0.3  # Fattore di probabilità di mutazione
p_lm = 50  # Livello di probabilità di mutazione
p_max_c = 0.8 # probabilità massima di crossover
p_min_c = 0.5 # probabilità minima di crossover
p_max_m = 0.3 # probabilità massima di mutazione
p_min_m = 0.1 # probabilità minima di mutazione
global Tmax; #calcoli massimi per la fitness
global t_fit; #numero di volte che è stata calcolata la fitness
Tmax = 1000000

def fitness(individual, weights):
    global t_fit
    t_fit += 1
    return np.sum(individual * weights)

def average_fitness(population, weights):
    return np.mean([fitness(ind, weights) for ind in population])

def initialize_population(n, L):
    return [np.random.randint(2, size=L) for _ in range(n)]

def optimize_parameters(H, epsilon, p_fc, p_lc, p_fm, p_lm ,p_c, p_m):
    if len(H) < v:
        return 0.5, 0.01

    Y = H[-v:]
    X = np.arange(1, v + 1)
    variance_x = (v**2-1)/12
    cov_xy = np.cov(X, Y)[0][1]
    alpha = cov_xy/variance_x
    print("alpha = ", alpha)

    delta_mutate = (p_max_m-p_min_m) / p_lm
    delta_crossover = (p_max_c-p_min_c) / p_lc

    if alpha > epsilon:
        p = p_m - delta_mutate
        new_p_m = np.clip(p, p_min_m, p_max_m)
        p = p_c - delta_crossover
        new_p_c = np.clip(p, p_min_c, p_max_c)

    elif alpha < -epsilon:
        p = p_m + delta_mutate
        new_p_m = np.clip(p, p_min_m, p_max_m)
        p = p_c + delta_crossover
        new_p_c = np.clip(p, p_min_c, p_max_c)
    
    elif alpha >= -epsilon and epsilon >= alpha:
        p = p_m + (delta_mutate / p_fm)
        new_p_m = np.clip(p, p_min_m,p_max_m)
        p = p_c + (delta_crossover / p_fc)
        new_p_c = np.clip(p, p_min_c, p_max_c)

    return new_p_c, new_p_m

def tournament_selection(population, weights):
    return [max(random.sample(population, 3), key=lambda ind: fitness(ind, weights)) for _ in range(len(population))]

def roulette_selection(population, weights):
    fitness_values = [fitness(ind, weights) for ind in population]
    total_fitness = sum(fitness_values)
    probabilities = [sum(fitness_values[:i+1]) / total_fitness for i in range(len(fitness_values))]
    
    selected = []
    for _ in range(len(population)):
        r = random.random()
        for i, individual in enumerate(population):
            if r <= probabilities[i]:
                selected.append(individual)
                break
    return selected

def elitism_selection(population, weights, elite_size=3):
    fitness_values = [(fitness(ind, weights), ind) for ind in population]
    sorted_population = sorted(fitness_values, key=lambda x: x[0], reverse=True)
    elites = [ind for _, ind in sorted_population[:elite_size]]
    non_elites = [ind for _, ind in sorted_population[elite_size:]]
    selected = elites + non_elites[:len(population) - elite_size]
    return selected

def rank_selection(population, weights):
    fitness_values = [(fitness(ind, weights), ind) for ind in population]
    sorted_population = sorted(fitness_values, key=lambda x: x[0])
    ranks = range(1, len(population) + 1)
    total_ranks = sum(ranks)
    probabilities = [sum(ranks[:i+1]) / total_ranks for i in range(len(ranks))]
    
    selected = []
    for _ in range(len(population)):
        r = random.random()
        for i, (_, individual) in enumerate(sorted_population):
            if r <= probabilities[i]:
                selected.append(individual)
                break
    return selected

def update_probabilities(probs, index, delta):
    min_prob = 0.3
    max_prob = 0.7
    n = len(probs)
    new_probs = probs[:]
    new_probs[index] += delta
    
    decrement = delta / (n - 1)
    for i in range(n):
        if i != index:
            new_probs[i] -= decrement
    
    excess = 0.0
    for i in range(n):
        if new_probs[i] < min_prob:
            excess += min_prob - new_probs[i]
            new_probs[i] = min_prob
        elif new_probs[i] > max_prob:
            excess += new_probs[i] - max_prob
            new_probs[i] = max_prob
    
    for i in range(n):
        if i != index:
            if new_probs[i] + excess / (n - 1) <= max_prob:
                new_probs[i] += excess / (n - 1)
                excess = 0.0
            else:
                excess -= (max_prob - new_probs[i]) * (n - 1)
                new_probs[i] = max_prob
    
    if excess > 0:
        for i in range(n):
            if new_probs[i] + excess / n <= max_prob:
                new_probs[i] += excess / n
            else:
                excess -= (max_prob - new_probs[i]) * n
                new_probs[i] = max_prob
    
    total = sum(new_probs)
    new_probs = [p / total for p in new_probs]

    return new_probs
    
def optimize_selection(probs, weights, H, epsilon, population, method):
    
    if len(H) >= v:
        Y = H[-v:]
        X = np.arange(1, v + 1)
        variance_x = (v**2-1)/12
        cov_xy = np.cov(X, Y)[0][1]
        alpha = cov_xy/variance_x
        delta = 0.4

        if alpha > epsilon:
            # significa che la fitness media è cresciuta, quindi aumenta il peso del metodo scelto in maniera proporzionale alla fitness media cresciuta, e decrementa la probabilità degli altri 3 metodi
            if(method=="tournament"):
                probs = update_probabilities(probs, 0, delta)
            elif(method=="roulette"):
                probs = update_probabilities(probs, 1, delta)
            elif(method=="elitism"):
                probs = update_probabilities(probs, 2, delta)
            elif(method=="rank"):
                probs = update_probabilities(probs, 3, delta)

        elif alpha < -epsilon:
            # significa che la fitness media è diminuita, quindi decrementa il peso del
            # metodo scelto in maniera proporzionale alla fitness media diminuita, e incrementa la probabilità degli altri 3 metodi
            if(method=="tournament"):
                probs = update_probabilities(probs, 0, -delta)
            elif(method=="roulette"):
                probs = update_probabilities(probs, 1, -delta)
            elif(method=="elitism"):
                probs = update_probabilities(probs, 2, -delta)
            elif(method=="rank"):
                probs = update_probabilities(probs, 3, -delta)
        
        elif alpha >= -epsilon and epsilon >= alpha:
            # significa che la fitness media è rimasta costante, quindi riporto i pesi ai valori originali
            probs = [0.25, 0.25, 0.25, 0.25]

    #scelgo il metodo di selezione in base alle probabilità
    r = random.random()
    if r < probs[0]:
        method = "tournament"
        return probs, method, tournament_selection(population, weights)
    elif r < probs[0] + probs[1]:
        method = "roulette"
        return probs, method, roulette_selection(population, weights)
    elif r < probs[0] + probs[1] + probs[2]:
        method = "elitism"
        return probs, method, elitism_selection(population, weights)
    else:
        method = "rank"
        return probs, method, rank_selection(population, weights)


def single_point_crossover(population, p_c):
    new_population = []
    for i in range(0, len(population), 2):
        if i+1 < len(population) and random.random() < p_c:
            crossover_point = random.randint(1, len(population[i]) - 1)
            new_population.append(np.concatenate((population[i][:crossover_point], population[i+1][crossover_point:])))
            new_population.append(np.concatenate((population[i+1][:crossover_point], population[i][crossover_point:])))
        else:
            new_population.append(population[i])
            if i+1 < len(population):
                new_population.append(population[i+1])
    return new_population

def mutation_operator(population, p_m):
    for individual in population:
        if random.random() < p_m:
            mutation_point = random.randint(0, len(individual) - 1)
            individual[mutation_point] = 1 - individual[mutation_point]
    return population

def create_run_directory():
    if not(os.path.exists("Instance_length_1500_weights_-25_to_25")):
        os.makedirs("Instance_length_1500_weights_-25_to_25")
    run_number = 1
    while os.path.exists(f"Instance_length_1500_weights_-25_to_25/run_{run_number}"):
        run_number += 1
    new_run_dir = f"Instance_length_1500_weights_-25_to_25/run_{run_number}"
    os.makedirs(new_run_dir)
    return new_run_dir

def read_weights_from_file(filename):
    with open(filename, 'r') as file:
        line = file.readlines()
        line = line[0]
        current_weights = []
        
        if line.startswith("Weights:"):
            line = line.replace("Weights:", "").strip()
            current_weights = [float(num) for num in line.split()]
    return current_weights


def selectionGA(k, n, v, epsilon, p_fc, p_lc, p_fm, p_lm,weights):
    global t_fit
    global Tmax
    tfit = 0
    t = 0
    H = []
    p_c = 0.5
    p_m = 0.01
    P = initialize_population(n, len(weights))
    avg_fit = average_fitness(P, weights)
    H.append(avg_fit)
    probs_selection = [0.25, 0.25, 0.25, 0.25] # p_tournament, p_roulette, p_elitism, p_rank
    method_selection = "tournament"
    crossover_rates = []
    mutation_rates = []
    fitnesses = []
    probs_selection_list = []
    tournament_counter = 0
    roulette_counter = 0
    elitism_counter = 0
    rank_counter = 0
    while t <= k and t_fit <= (Tmax-n):
        probs_selection, method_selection, P_star = optimize_selection(probs_selection, weights, H, epsilon, P, method_selection)
        probs_selection_list.append(probs_selection)
        if method_selection == "tournament":
            tournament_counter += 1
        elif method_selection == "roulette":
            roulette_counter += 1
        elif method_selection == "elitism":
            elitism_counter += 1
        elif method_selection == "rank":
            rank_counter += 1
        p_c, p_m = optimize_parameters(H, epsilon, p_fc, p_lc, p_fm, p_lm, p_c , p_m)
        crossover_rates.append(p_c)
        mutation_rates.append(p_m)
        P_star = single_point_crossover(P_star, p_c)
        P_star = mutation_operator(P_star, p_m)
        avg_fit = average_fitness(P_star, weights)
        fitnesses.append(avg_fit)
        H.append(avg_fit)
        if len(H) > v:
            H.pop(0)
        P = P_star
        t += 1
        print("t=", t, " Selection: ",method_selection," AVG_Fitness:", avg_fit, "Crossover rate:", p_c, "Mutation rate:", p_m)

    if t_fit > Tmax:
        print(f"Numero massimo di calcoli della fitness ({Tmax}) raggiunto.")

    print("Fitness calcolata ", t_fit, " volte.")

    best_individual = max(P, key=lambda ind: fitness(ind, weights))
    return best_individual, fitnesses, crossover_rates, mutation_rates, probs_selection_list, tournament_counter, roulette_counter, elitism_counter, rank_counter

filename = "C:\\Users\\smbsv\\Desktop\\istanze\\Instance_length_1500_weights_-25_to_25.txt"
weights = read_weights_from_file(filename)
print("Lista di array di pesi:")
print(weights)
print("...........................................................................................")

all_fitness = []
times = []
start_time = time.time()  
for _ in range(50):
    random.seed(datetime.now().timestamp())  # reset del seed per ogni run
    t_fit = 0
    weights = np.array(weights)

    current_time = time.time()
    best_solution, fitnesses, crossover_rates, mutation_rates, probs_selection_list, tournament_counter, roulette_counter, elitism_counter, rank_counter = selectionGA(k, n, v, epsilon, p_fc, p_lc, p_fm, p_lm, weights)
    times.append(current_time-start_time)

    print("Soluzione ottima globalmente: ", np.sum(weights[weights > 0]))
    print("Fitness della migliore soluzione:", fitness(best_solution, weights))

    all_fitness.append(fitness(best_solution, weights))

    run_dir = create_run_directory()

    # grafico dell'andamento delle probabilità dei metodi scelti
    plt.figure(figsize=(12, 8))
    probs_selection_list = probs_selection_list[0:200] #filtriamo solo per le prime 200 iterazioni
    prob1 = [item[0] for item in probs_selection_list]
    prob2 = [item[1] for item in probs_selection_list]
    prob3 = [item[2] for item in probs_selection_list]
    prob4 = [item[3] for item in probs_selection_list]
    plt.figure(figsize=(24, 6))
    plt.plot(prob1, label='Tournament')
    plt.plot(prob2, label='Roulette')
    plt.plot(prob3, label='Elitism')
    plt.plot(prob4, label='Rank')
    plt.title('Andamento delle Probabilità')
    plt.xlabel('Iterazione')
    plt.ylabel('Probabilità')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'selection_probability.png'))
    plt.close()

    # Istogramma col numero di volte che è stato scelto un metodo di selezione
    plt.figure(figsize=(12, 8))
    methods = ['Tournament', 'Roulette', 'Elitism', 'Rank']
    counts = [tournament_counter, roulette_counter, elitism_counter, rank_counter]
    plt.bar(methods, counts)
    plt.title('#Selezioni per metodo')
    plt.xlabel('Metodo')
    plt.ylabel('#Selezioni')
    plt.savefig(os.path.join(run_dir, 'selection_histogram.png'))
    plt.close()

    # Grafico della fitness media
    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    plt.plot(range(1, len(fitnesses)+1), fitnesses, linestyle='-', color='b')
    plt.axhline(y=np.sum(weights[weights > 0]), color='r', linestyle='--', label='Soluzione Ottima: ' + str(np.sum(weights[weights > 0])))
    plt.axhline(y=fitness(best_solution, weights), color='g', linestyle='--', label='Soluzione migliore: ' + str(fitness(best_solution, weights)))
    
    plt.title('Andamento della fitness media della popolazione')
    plt.xlabel('Generazioni')
    plt.ylabel('Fitness media')
    plt.legend()
    current_ylim = plt.ylim()
    new_ylim = (current_ylim[0], current_ylim[1] * 1.1)
    plt.ylim(new_ylim)
    plt.savefig(os.path.join(run_dir, 'fitness_media.png'))
    plt.close()

    # Grafico del crossover rate
    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    plt.plot(range(1, len(crossover_rates)+1), crossover_rates, linestyle='-', color='r')
    plt.title('Andamento Crossover rate')
    plt.xlabel('Generazioni')
    plt.ylabel('Crossover Rate')
    plt.savefig(os.path.join(run_dir, 'crossover_rate.png'))
    plt.close()

    # Grafico del mutation rate
    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    plt.plot(range(1, len(mutation_rates)+1), mutation_rates, linestyle='-', color='g')
    plt.title('Andamento Mutation rate')
    plt.xlabel('Generazioni')
    plt.ylabel('Mutation Rate')
    plt.savefig(os.path.join(run_dir, 'mutation_rate.png'))
    plt.close()

    #scrivo un file di testo con la migliore soluzione trovata in questa run 
    with open(os.path.join(run_dir, 'found_solution_info.txt'), 'w') as file:
        file.write("[ "+' '.join(map(str, best_solution))+" ]")
        file.write("\nFitness: "+str(fitness(best_solution, weights)))
        if(t_fit > Tmax):
            file.write("\nNumero massimo di calcoli della fitness raggiunto")
        file.write("\n#Selezioni torneo: "+str(tournament_counter))
        file.write("\n#Selezioni roulette: "+str(roulette_counter))
        file.write("\n#Selezioni elitism: "+str(elitism_counter))
        file.write("\n#Selezioni rank: "+str(rank_counter))

    print("\nGrafici e miglior soluzione salvati nella cartella:", run_dir)


#scrittura dei parametri usati e info sulla fitness delle diverse soluzioni in un file di testo
with open(os.path.join(create_run_directory()+"/..", 'all_runs_info.txt'), 'w') as file:
        file.write(str("Parametri relativi a 50 run sulle istanze 'Instance_length_1500_weights_-25_to_25'\n"))
        file.write(str("#iterazioni="+ str(k) + "\n"))
        file.write(str("#popSize="+ str(n) + "\n"))
        file.write(str("DimRegressionBuffer="+ str(v) + "\n"))
        file.write(str("epsilon="+ str(epsilon) + "\n"))
        file.write(str("Delta=0.4\n"))
        file.write(str("p_fc="+ str(p_fc) + "\n"))
        file.write(str("p_lc="+ str(p_lc) + "\n"))
        file.write(str("p_fm="+ str(p_fm) + "\n"))
        file.write(str("p_lm="+ str(p_lm) + "\n"))
        file.write(str("p_max_c="+ str(p_max_c) + "\n"))
        file.write(str("p_min_c="+ str(p_min_c) + "\n"))
        file.write(str("p_max_m="+ str(p_max_m) + "\n"))
        file.write(str("p_min_m="+ str(p_min_m) + "\n"))
        file.write(str("Tmax="+ str(Tmax) + "\n"))
        file.write(str("Candidati al torneo (in caso di tournament selection)=3\n"))
        file.write(str("Fitness ottima globale="+str(np.sum(weights[weights > 0]))+"\n"))
        file.write(str("Soluzione con fitness massima della run="+str(max(all_fitness))+"\n"))
        file.write(str("Media delle fitness delle migliori soluzioni della run="+str(np.mean(all_fitness))+"\n"))
        file.write(str("Deviazione standard delle fitness delle migliori soluzioni della run="+str(np.std(all_fitness))+"\n"))

# Grafico delle soluzioni delle run (migliore,media, dev. standard) in base alle generazioni
plt.figure(figsize=(16, 8))
plt.tight_layout()
plt.scatter(range(len(all_fitness)), all_fitness, label='Valori di fitness nelle 50 run ')
plt.title('Info fitness nelle 50 run. Istanza "Instance_length_1500_weights_-25_to_25"')
plt.xlabel('Run')
plt.ylabel('Fitness')
plt.axhline(y=np.mean(all_fitness), color='r', linestyle='--', label='Media delle fitness: ' + str(np.mean(all_fitness)))
# linee che indicano la  deviazione standard
plt.axhline(y=np.mean(all_fitness)-np.std(all_fitness), color='g', linestyle='--', label='Dev Standard: ' + str(np.std(all_fitness)))
plt.axhline(y=np.mean(all_fitness)+np.std(all_fitness), color='g', linestyle='--')
plt.legend()

plt.savefig(os.path.join(create_run_directory()+"/..", 'generations_fitnesses_scatter.png'))
plt.close()

# Grafico delle soluzioni delle run (migliore,media, dev. standard) in base al tempo
plt.figure(figsize=(16, 8))
plt.tight_layout()
plt.scatter(times, all_fitness, label='Valori di fitness nel tempo ')
plt.title('Info fitness nelle 50 run. Istanza "Instance_length_1500_weights_-25_to_25"')
plt.xlabel('Time (s)')
plt.ylabel('Fitness')
plt.axhline(y=np.mean(all_fitness), color='r', linestyle='--', label='Media delle fitness: ' + str(np.mean(all_fitness)))
# linee che indicano la  deviazione standard
plt.axhline(y=np.mean(all_fitness)-np.std(all_fitness), color='g', linestyle='--', label='Dev Standard: ' + str(np.std(all_fitness)))
plt.axhline(y=np.mean(all_fitness)+np.std(all_fitness), color='g', linestyle='--')
plt.legend()

plt.savefig(os.path.join(create_run_directory()+"/..", 'time_fitnesses_scatter.png'))
plt.close()