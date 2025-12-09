from random import randint, uniform
import numpy as np
from EA_problem import Problem

class Knapsack(Problem):
    def __init__(self, path):
        self._data = list()

        file = open(path, "r")
        # max items and knapsack capacity
        parameters = file.readline().strip().split()
        self._max_items = int(parameters[0])
        self._max_capacity = int(parameters[1])
        
        # dataset : (profit, weight)
        for line in file:
            val = line.strip().split(' ')
            self._data.append((int(val[0]), int(val[1])))
    
    def populationInitialization(self, pop_size):
        population = np.random.randint(2, size = (pop_size, self._max_items))
        population = population.tolist()
        return population

    def fitnessFunction(self, population):
        # fitness value is the sum of profits, and zero if items in a chromosome outweight max knapsack weight
        fitness = np.zeros(len(population)).tolist()
        for i in range(len(population)):
            total_profit = 0
            total_weight = 0
            for j in range(self._max_items):
                total_profit += population[i][j] * self._data[j][0]
                total_weight += population[i][j] * self._data[j][1]
            if total_weight <= self._max_capacity:
                fitness[i] = total_profit
            else:
                # not a valid solution
                fitness[i] = 0 
        return fitness
    
    def crossOver(self, parentA, parentB):
        offspring = parentA[:len(parentA)//2] + parentB[len(parentB)//2:]
        return offspring
    
    def mutation(self, chromosome, mutation_rate):
        if uniform(0, 1) < mutation_rate:
            random_pos = randint(0, len(chromosome)-1)
            if chromosome[random_pos] == 0:
                chromosome[random_pos] = 1
            else:
                chromosome[random_pos] = 0
        return chromosome
