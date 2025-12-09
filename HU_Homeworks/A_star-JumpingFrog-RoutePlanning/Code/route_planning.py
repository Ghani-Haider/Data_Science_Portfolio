import pandas as pd
from astar_search import *

# add path to csv files
CITIES = "CSV\cities.csv"
CONNECTION = "CSV\Connections.csv"
HEURISTICS = "CSV\heuristics.csv"

class RoutePlanning(SearchProblem):
    def __init__(self, start_city, destination_city) -> None:
        super().__init__()

        # start and goal states
        self.startstate = start_city
        self.goalstate = destination_city

        # creating dataframe from csv files
        self.cities_df = pd.read_csv(CITIES)
        self.connection_df = pd.read_csv(CONNECTION)
        self.heuristics_df = pd.read_csv(HEURISTICS)

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        return self.startstate

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        return state == self.goalstate

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        successor_lst = []
        for value in range(len(self.connection_df[state])):
            if (self.connection_df[state][value] != 0 and self.connection_df[state][value] != -1):
                successor = self.connection_df.iloc[value][0]
                action = value
                stepCost = self.connection_df[state][value]
                successor_lst.append((successor, action, stepCost))
        return successor_lst

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

    def getHeuristic(self, state):
        """
        state: the current state of agent

        This function returns the heuristic of current state of the agent which will be the
        estimated distance from goal.
        """
        for value in range(len(self.heuristics_df[state])):
            if self.heuristics_df.iloc[value][0] == self.goalstate:
                return self.heuristics_df[state][value]


def main():
    problem = RoutePlanning("Taxila", "Khunjerab Pass") #(start, destination)
    lst = aStarSearch(problem)
    print(lst)

if __name__ == "__main__":
    main()
