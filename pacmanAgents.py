# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import scoreEvaluation
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        print("Legal",legal)
        # get all the successor state for these actions
        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        print(successors)
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        print(scored)
        # get best choice
        bestScore = max(scored)[0]
        
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        print(bestActions)
        print("")
        print("")
        # return random action from the list of the best actions
        return random.choice(bestActions)
    
    
class BFSAgent(Agent):
    
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;
    

    # GetAction Function: Called with every frame
    def getAction(self, state):
    # TODO: write DFS Algorithm instead of returning Directions.STOP
        node_stack = []
        leaf_nodes = []
        nodes = {}
        nodes["state"] = state
        nodes["action"] = None
        nodes["ancestor"] = None
    
        legal = state.getLegalPacmanActions()
        random.shuffle(legal)
        successor = [(state.generatePacmanSuccessor(action),action) for action in legal]
        
        for element in successor:
            temp_nodes = {}
            temp_nodes["state"] =element[0]
            temp_nodes["action"] = element[1]
            temp_nodes["ancestor"] = nodes
            node_stack.append(temp_nodes)

        while node_stack:        
            current_node = node_stack.pop(0)
            i_state = current_node["state"]
            i_action = current_node["action"]
            
            if (i_state is not None):
                legal = i_state.getLegalPacmanActions()
                #random.shuffle(legal)
                if legal:
                    successor = [(i_state.generatePacmanSuccessor(action),i_action) for action in legal]        
               
                refined_successor = [element for element in successor if None not in element]
                
                if (i_state.isWin()) or (i_state.isLose()) or (refined_successor is None):
                    leaf_nodes.append(current_node)
                else:
                    for successor_child in refined_successor:
                        temp_nodes ={}
                        temp_nodes["state"] = successor_child[0]
                        temp_nodes["action"] = successor_child[1]
                        temp_nodes["ancestor"] = nodes
                        leaf_nodes.append(temp_nodes)

        max_score = float("-inf")
        if leaf_nodes is not None:
            for j in leaf_nodes:
                current_score = scoreEvaluation(j["state"])
                if(current_score >= max_score):
                    max_score = current_score
                    final_action = j["action"]
            return final_action
        
class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
    # TODO: write DFS Algorithm instead of returning Directions.STOP    
        node_stack = []
        leaf_nodes = []
        nodes = {}
        nodes["state"] = state
        nodes["action"] = None
        nodes["ancestor"] = None
    
        legal = state.getLegalPacmanActions()
        random.shuffle(legal)
        successor = [(state.generatePacmanSuccessor(action),action) for action in legal]
        
        for element in successor:
            temp_nodes = {}
            temp_nodes["state"] =element[0]
            temp_nodes["action"] = element[1]
            temp_nodes["ancestor"] = nodes
            node_stack.append(temp_nodes)

        while node_stack:        
            current_node = node_stack.pop()
            i_state = current_node["state"]
            i_action = current_node["action"]
            
            if (i_state is not None):
                legal = i_state.getLegalPacmanActions()
                #random.shuffle(legal)
                if legal:
                    successor = [(i_state.generatePacmanSuccessor(action),i_action) for action in legal]        
               
                refined_successor = [element for element in successor if None not in element]
                
                if (i_state.isWin()) or (i_state.isLose()) or (refined_successor is None):
                    leaf_nodes.append(current_node)
                else:
                    for successor_child in refined_successor:
                        temp_nodes ={}
                        temp_nodes["state"] = successor_child[0]
                        temp_nodes["action"] = successor_child[1]
                        temp_nodes["ancestor"] = nodes
                        node_stack.append(temp_nodes)

        max_score = float("-inf")
        if leaf_nodes is not None:
            for j in leaf_nodes:
                current_score = scoreEvaluation(j["state"])
                if(current_score >= max_score):
                    max_score = current_score
                    final_action = j["action"]
    
            return final_action

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        node_stack = []
        leaf_nodes = []
        nodes = {}
        nodes["state"] = state
        nodes["action"] = None
        nodes["ancestor"] = None
        nodes["depth"] = None
        nodes["h(x)"] = None
        nodes["total_cost"] = None
        
        original_state = state
        legal = state.getLegalPacmanActions()
        successor = [(state.generatePacmanSuccessor(action),action) for action in legal]

        for element in successor:
            temp_nodes = {}
            temp_nodes["state"] = element[0]
            temp_nodes["action"] = element[1]
            temp_nodes["ancestor"] = state
            temp_nodes["depth"] = 1
            temp_nodes["h(x)"] = scoreEvaluation(original_state) - scoreEvaluation(temp_nodes["state"])
            temp_nodes["total_cost"] = temp_nodes["depth"] - temp_nodes["h(x)"]
            node_stack.append(temp_nodes)

        while node_stack:
            node_stack = sorted(node_stack, key=lambda k: k['total_cost'])
            current_node = node_stack.pop(0)
            i_state = current_node["state"]
            i_action = current_node["action"]
            i_depth = current_node["depth"]               
            
            legal = i_state.getLegalPacmanActions()
            if legal:
                successor = [(i_state.generatePacmanSuccessor(action),i_action) for action in legal]        
               
            refined_successor = [element for element in successor if None not in element]

            if (i_state.isWin()) or (i_state.isLose()) or (refined_successor is None):
                leaf_nodes.append(current_node)
            else:
                for successor_child in refined_successor:
                    if (successor_child[0] is not None):
                        temp_nodes = {}
                        temp_nodes["state"] = successor_child[0]
                        temp_nodes["action"] = successor_child[1]
                        temp_nodes["ancestor"] = current_node
                        temp_nodes["depth"] = current_node["depth"] + 1
                        temp_nodes["h(x)"] = scoreEvaluation(original_state) - scoreEvaluation(successor_child[0])
                        temp_nodes["total_cost"] = temp_nodes["depth"] + temp_nodes["h(x)"]
                        node_stack.append(temp_nodes)
                        
        max_score = float("-inf")
        final_action = None
        if leaf_nodes is not None:        
            for j in leaf_nodes:
                current_score = scoreEvaluation(j["state"])
                if(current_score >= max_score):
                    max_score = current_score
                    final_action = j["action"]
            return final_action
