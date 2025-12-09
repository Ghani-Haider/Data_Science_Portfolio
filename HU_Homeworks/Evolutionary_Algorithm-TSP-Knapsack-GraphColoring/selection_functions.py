import random
import numpy as np

def fitness_proportional(population, fitness_values, no_of_selection, max_prob):
    sum_of_fitness = 0
    cumulative_fitness = 0
    fitness_values_cumulative = list()
    if max_prob == True:
        sum_of_fitness = sum(fitness_values)
        for i in range(len(population)):
            cumulative_fitness += (fitness_values[i] / sum_of_fitness)
            fitness_values_cumulative.append(cumulative_fitness)
    else:
        for fitness in fitness_values:
            sum_of_fitness += 1 / (fitness + 1)
        for i in range(len(population)):
            cumulative_fitness += ((1 / (fitness_values[i] + 1)) / sum_of_fitness)
            fitness_values_cumulative.append(cumulative_fitness)
    # calculate best chromosome
    new_population = list()
    for i in range(no_of_selection):
        random_val = random.uniform(0,1)
        for j in range(len(fitness_values_cumulative)):
            if (random_val <= fitness_values_cumulative[j]):
                new_population.append(population[j])
                break
    return new_population

def rank_based(population, fitness_values, no_of_selection, max_prob):
    temp_fitness = list(fitness_values)
    total_fitness = 0
    ranks = np.zeros(len(temp_fitness), dtype=int).tolist()
    for i in range(len(temp_fitness)):
        val = max(temp_fitness) if max_prob == True else min(temp_fitness)
        ranks[temp_fitness.index(val)] = len(temp_fitness) - (i)
        total_fitness += (i + 1)
        temp_fitness[temp_fitness.index(val)] = -9999999999 if max_prob == True else 9999999999
    # fitness proportion based on ranks
    for i in range(len(temp_fitness)):
        ranks[i] = ranks[i] / total_fitness
    # cumulative
    for i in range(1, len(temp_fitness)):
        ranks[i] = ranks[i] + ranks[i-1]
    # calculate best chromosome
    new_population = list()
    for i in range(no_of_selection):
        random_val = random.uniform(0,1)
        for j in range(len(ranks)):
            if (random_val <= ranks[j]):
                new_population.append(population[j])
                break
    return new_population

def binary_tournament(population, fitness_values, no_of_selection, max_prob):
    new_population = list()
    for i in range(no_of_selection):
        random_val_x = random.randint(0, len(population)-1)
        chromosome_x = population[random_val_x]
        random_val_y = random.randint(0, len(population)-1)
        chromosome_y = population[random_val_y]
        if fitness_values[random_val_x] > fitness_values[random_val_y]:
            if max_prob:
                new_population.append(chromosome_x)
            else:
                new_population.append(chromosome_y)
        else:
            if max_prob:
                new_population.append(chromosome_y)
            else:
                new_population.append(chromosome_x)
    return new_population

def truncation(population, fitness_values, no_of_selection, max_prob):
    temp = fitness_values.copy()
    new_population = list()
    for i in range(no_of_selection):
        best_chromosome = max(temp) if max_prob == True else min(temp)
        new_population.append(population[temp.index(best_chromosome)])
        temp[temp.index(best_chromosome)] = -9999999999 if max_prob == True else 9999999999
    return new_population

def random_selection(population, fitness_values, no_of_selection, max_prob):
    new_population = list()
    temp = population.copy()
    for i in range(no_of_selection):
        random_val = random.randint(0, len(temp)-1)
        chromosome = temp[random_val]
        new_population.append(chromosome)
    return new_population

def selection_function(population, fitness_values, no_of_selection, max_prob, selection_method):
    if selection_method == "FPS":
        return fitness_proportional(population, fitness_values, no_of_selection, max_prob)
    elif selection_method == "RBS":
        return rank_based(population, fitness_values, no_of_selection, max_prob)
    elif selection_method == "Binary Tournament":
        return binary_tournament(population, fitness_values, no_of_selection, max_prob)
    elif selection_method == "Truncation":
        return truncation(population, fitness_values, no_of_selection, max_prob)
    elif selection_method == "Random":
        return random_selection(population, fitness_values, no_of_selection, max_prob)