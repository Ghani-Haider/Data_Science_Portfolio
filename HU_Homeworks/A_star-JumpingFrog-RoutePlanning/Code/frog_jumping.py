from astar_search import *


class JumpingFrogProblem(SearchProblem):
    def __init__(self):
        self.StartState = "GGG*RRR"
        self.GoalState = "RRR*GGG"
        self.currentState = self.StartState
        self.costSoFar = 0
        self.vacantRock = 3

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        # util.raiseNotDefined()
        return self.StartState

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        if state == self.GoalState:
            return True
        return False

        util.raiseNotDefined()

    def getGoalState(self):
        return self.GoalState

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        successors = []
        emptyPos = state.index("*")
        for i in range(len(state)):
            if state[i] == '*':
                emptyPos = i
                break
        for i in range(len(state)):
            if i - emptyPos == 1 and state[i] == 'R':
                x = state[:emptyPos]
                x += state[i] + "*"
                x += state[i + 1:]
                successors.append((x, "JumpOne", 1))
            elif emptyPos - i == 1 and state[i] == 'G':
                x = state[:i]
                x += "*"+state[i]
                x = x+state[emptyPos + 1:]
                successors.append((x, "JumpOne", 1))
            elif emptyPos - i == 2 and state[i] == 'G':
                x = state[:i]
                x += "*" + state[i + 1]+state[i]
                x += state[emptyPos + 1:]
                successors.append((x, "JumpTwo", 2))
            elif i - emptyPos == 2 and state[i] == 'R':
                x = state[:emptyPos]
                x += state[i] + state[emptyPos + 1] + "*"
                x += state[i + 1:]

                successors.append((x, "JumpTwo", 2))

        return successors

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        pass

    def getHeuristic(self, state):
        """
        state: the current state of agent
        THis function returns the heuristic of current state of the agent which will be the
        estimated distance from goal.
        """
        score = 7
        for i in range(len(state)):
            if state[i] == self.GoalState[i]:
                score -= 1
        return score

def main():
    problem = JumpingFrogProblem()
    lst = aStarSearch(problem)
    print(lst)

if __name__ == "__main__":
    main()

