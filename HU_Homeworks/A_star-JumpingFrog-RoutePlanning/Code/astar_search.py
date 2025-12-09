# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from os import close
from util import PriorityQueue
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """

        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """

        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

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

        THis function returns the heuristic of current state of the agent which will be the
        estimated distance from goal.
        """
        util.raiseNotDefined()


def aStarSearch(problem: SearchProblem):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR A* CODE HERE ***"
    # open and close lists
    open_list = PriorityQueue()
    open_list_check = []
    closed = set()

    # get and add start state to open_list
    start_state = problem.getStartState()
    open_list.push(start_state, 0)
    open_list_check.append(start_state)

    # for backtracking
    parent = dict()
    parent[start_state] = None
    # for f(state)
    best_cost_f = dict()
    best_cost_f[start_state] = 0
    # for g(state)
    best_cost_g = dict()
    best_cost_g[start_state] = 0
    
    flag = False
    while (open_list.isEmpty() != True and flag == False):
        current_state = open_list.pop()
        open_list_check.remove(current_state)

        successors = problem.getSuccessors(current_state)
        for successor, action, stepCost in successors:
            if (problem.isGoalState(successor)):
                flag = True
                parent[successor] = current_state
                break
            
            # calculating g(successor) and f(successor)
            successor_cost_so_far = best_cost_g[current_state] + stepCost
            successor_f = successor_cost_so_far + problem.getHeuristic(successor)

            # only add/update the successor to open if its not added before OR new f() is lower
            if successor in open_list_check and successor_f > best_cost_f[successor]:
                continue
            elif successor in closed and successor_f > best_cost_f[successor]:
                continue
            else:
                parent[successor] = current_state
                open_list.push(successor, successor_f)
                open_list_check.append(successor)
                best_cost_f[successor] = successor_f
                best_cost_g[successor] = successor_cost_so_far
        closed.add(current_state)

    # backtraking
    path = []
    start = problem.getStartState()
    for p in parent:
        if problem.isGoalState(p) == True:
            current = p
            break
    while current != start:
        path.insert(0, current)
        current = parent[current]
    path.insert(0, start)
    return path
