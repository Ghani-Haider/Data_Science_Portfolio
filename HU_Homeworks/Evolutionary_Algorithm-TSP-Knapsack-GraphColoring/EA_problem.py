class Problem:
    def __init__():
        pass

    def populationInitialization(self, pop_size):
        """
            pop_size: Size of the population to be initialize.
        
        Initialize the population randomly or with potentially good solutions.
        """
        pass

    def fitnessFunction(self, population):
        """
            Population: A list containing chromosomes

        Returns fitness values of the chromosomes in the population.
        """
        pass

    def crossOver(self, parentA, parentB):
        """
            parentA: Parent chromosome
            parentB: Parent chromosome

        For given parents, this should return a new offspring based on infomration 
        sharing between parent according to some criteria.
        """
        pass

    def mutation(self, chromosome, mutation_rate):
        """
            chromosome: An individual in the population
            mutation_rate: Probability of chromosome being mutated

        Unitary operation that allows random change in offspring and return mutated offsrping.
        """
        pass