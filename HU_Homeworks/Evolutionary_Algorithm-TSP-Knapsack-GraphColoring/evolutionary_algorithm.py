import matplotlib.pyplot as plt
from EA_problem import *
from selection_functions import *
from knapsack import Knapsack
from graphcoloring import Graphcoloring
from tsp import tsp

population_size = 50
no_of_offspring = 30
no_of_generations = 800
mutation_rate = 0.3
no_of_iterations = 5

def evolutionaryAlgorithm(problem: Problem, max_problem, parent_selection, survivor_selection):
    generation_BSF = [0] * no_of_generations
    generation_ASF = [0] * no_of_generations
    
    #* Step 1: Initialize the population randomly or with potentially good solutions
    # list of chromosomes
    population = problem.populationInitialization(population_size)
    
    for j in range(no_of_iterations):
        #* Step 7: Go to Step 2 until the termination criteria are met
        for i in range(no_of_generations):

            #* Step 2: Compute the fitness of each individual in the population
            # list of fitness values corresponding to population
            fitness_values = problem.fitnessFunction(population)
            # calculating sum of avg. and best fitness so far for each generation
            if max_problem:
                generation_BSF[i] = generation_BSF[i] + max(fitness_values)
            else:
                generation_BSF[i] = generation_BSF[i] + min(fitness_values)
            generation_ASF[i] = generation_ASF[i] + sum(fitness_values)/len(fitness_values)

            #* Step 3: Select parents using a selection procedure
            # double times the no_of_offsprings
            parent_list = selection_function(population, fitness_values, 2*no_of_offspring, max_problem, parent_selection)

            #* Step 4: Create offspring by crossover and mutation operators
            offspring_list = []
            for i in range(no_of_offspring-1):
                # corssover
                offspring = problem.crossOver(parent_list[i],
                            parent_list[i+1])
                # mutation
                offspring = problem.mutation(offspring, mutation_rate)
                offspring_list.append(offspring)

            #* Step 5 Compute the fitness of the new offspring
            offspring_fitness_val = problem.fitnessFunction(offspring_list)
            
            population = population + offspring_list
            fitness_values = fitness_values + offspring_fitness_val
            
            #* Step 6  Select members of population to survive using a selection procedure.
            population = selection_function(population, fitness_values, population_size, max_problem, survivor_selection)
    
    # calculating and plotting average best so far and average average fitnesss of each generation
    avg_generation_BSF = [x/no_of_iterations for x in generation_BSF]
    avg_generation_ASF = [x/no_of_iterations for x in generation_ASF]
    plt.plot(range(no_of_generations), avg_generation_BSF)
    plt.plot(range(no_of_generations), avg_generation_ASF)
    plt.legend(["Avg BSF", "Avg ASF"])
    plt.title("Parent Selection:" + parent_selection + " | Survivor Selection: " + survivor_selection, fontdict = {'fontsize' : 12})
    plt.xlabel("Generation", fontdict={'fontsize' : 13})
    plt.ylabel("Fitness", fontdict={'fontsize' : 13})
    plt.show()

    fitness_values = problem.fitnessFunction(population)
    return population, fitness_values

# ========================================================================================#
def main():
    ## max_problem = true if maximization problem, and false otherwise
    ## max_problem = true for knapsack, false for tsp and graph coloring
    maximization_problem = False

    ## set the problem to be solved
    # prob = Knapsack("f2_l-d_kp_20_878")
    prob = Graphcoloring("gcol1.txt")
    # prob = tsp("qa194.tsp")

    selection_scheme = ["FPS", "RBS", "Binary Tournament", "Truncation", "Random"]

    pop, fitness = evolutionaryAlgorithm(prob, maximization_problem, "Random", "Truncation")
    
    if maximization_problem:
        print('The best chromosome is {}'.format(pop[fitness.index(max(fitness))]))
        print('The best fitness value is {}'.format(max(fitness)))
    else:
        print('The best chromosome is {}'.format(pop[fitness.index(min(fitness))]))
        print('The best fitness value is {}'.format(min(fitness)))


if __name__ == "__main__":
    main()
