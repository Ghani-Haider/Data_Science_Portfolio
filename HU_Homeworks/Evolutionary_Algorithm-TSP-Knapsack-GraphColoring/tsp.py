import tsplib95
from random import randint, uniform
from EA_problem import Problem

class tsp(Problem):
    def __init__(self, path):
        self.tspdata = tsplib95.load(path)
        self.no_of_nodes = len(list(self.tspdata.get_nodes()))

    def populationInitialization(self, pop_size):
        init_pop = 0
        population = list()

        while init_pop != pop_size:
            node = randint(1, self.no_of_nodes)
            path = list()
            while len(path) != self.no_of_nodes:
                node = randint(1, self.no_of_nodes)
                if node not in path:
                    path.append(node)

            population.append(path)
            init_pop += 1

        return population

    def fitnessFunction(self, population):
        # fitness is the cummulative distance travelled
        fitness = [None] * len(population)
        for i in range(len(population)):
            total_weight = 0

            #weight of one path
            path=population[i]
            for j in range(1, len(path)):
                start = path[j-1]
                end = path[j]
                edge = start, end                       
                total_weight += self.tspdata.get_weight(*edge)
            fitness[i] = total_weight

        return fitness

    def crossOver(self, parentA, parentB):
        offspring=[None] * len(parentA)

        #select half of parentA to be carried forward
        start=randint(1,(len(parentA)//2)-2)    
        end=start+(len(parentA)//2)        
        offspring[start:end] = parentA[start:end]

        #update the rest of offspring via parentB
        parentb_idx=end
        offspring_idx=end
        while None in offspring:
            if parentB[parentb_idx] not in offspring:
                offspring[offspring_idx]=parentB[parentb_idx]
                offspring_idx=(offspring_idx+1)%(len(parentB))
            parentb_idx=(parentb_idx+1)%(len(parentB))

        return offspring

    def mutation(self, chromosome, mutation_rate):
        #if mutation occurs then swap too genes
        if uniform(0, 1) < mutation_rate:
            while True:
                pos1 = randint(1, len(chromosome)-2)
                pos2 = randint(1, len(chromosome)-2)
                if pos1!=pos2:
                    chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
                    break

        return chromosome
