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
from heuristics import *
import random
import math

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();

        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];

        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)



###############################################################################
#
#   Hill Climber Starts here.
#
###############################################################################
"""
Function : climb(actionLists, state, flag)

This is the evaluation function that gives the evaluation of the action sequence by
following those actions and reach a state and evaluate the score of that state.

This function returns 0 score and flag = False in case generatePacmanSuccessor gives None.
This situation happens when we run out of the maximum number of function calls allowed
to that method.

Input : actionLists : sequence of action that is supposed to be executed.
        state : the root state (current state of pacman)
        flag : the flag monitors the max allowed function calls.
        
Output : scoreEvaluation() : score of the generated state
         flag : value of flag (True or False)
"""
def climb(actionLists, state, flag):
    tempFunState = state
    for i in range(0,len(actionLists)):
        if tempFunState.isWin() + tempFunState.isLose() == 0:
            tempFunState = tempFunState.generatePacmanSuccessor(actionLists[i])
            if tempFunState is None:
                flag = False
                return 0, flag
        else:
            flag = False
            break    
    return scoreEvaluation(tempFunState), flag

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,5):
            self.actionList.append(Directions.STOP)       
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        #initializing some data
        finalScore = -500
        currentScore = float("-inf")
        finalAction = Directions.STOP
        flag = True
        
        #getting the possible values to initialize actionList
        possible = state.getAllPossibleActions()
        for i in range(0, len(self.actionList)):
            self.actionList[i] = possible[random.randint(0, len(possible)-1)]

        #compute scores
        currentScore, flag = climb(self.actionList, state, flag)
        finalScore = currentScore
        finalAction = self.actionList[0]

        #this loop finds the max valued neighbour
        while(True):
            # get all actions for pacman
            for i in range(0, len(self.actionList)):
                if random.randint(0,1) == 1:
                    self.actionList[i] = possible[random.randint(0, len(possible)-1)]

            #compute the score and updated value of flag
            currentScore, flag = climb(self.actionList, state, flag)
     
            #update the max values
            if(finalScore <= currentScore):
                finalScore = currentScore
                finalAction = self.actionList[0]

            #break the loop if we ran out of computational budget.
            if flag == False :
                break
        
        #return the action if it is in the legal pacman actions
        legalActions = state.getLegalPacmanActions()
        if finalAction not in legalActions:
            finalAction = Directions.STOP        
        
        #returns the final action
        return finalAction;


    
###############################################################################
#
#   Genetic Algorithm starts here.
#
###############################################################################
"""
Function : evalFun(actionLists, state)

This function evaluates the sequence of action by following the actions and 
evaluating the score of the final state achieved by that sequence.

It also updates the flag value in case we ran out of maximum number of function
calls allowed on generatePacmanSuccessor. 

Input : actionLists : sequence of actions that needs to be evaluated.
        state : current state of the pacman
        
Output : scoreEvaluation() : score of generated state
         flag : value of flag (True/False)
"""
def evalFun(actionLists, state):
    tempFunState = state
    for i in range(0,len(actionLists)):
        if tempFunState.isWin() + tempFunState.isLose() == 0:
            tempFunState = tempFunState.generatePacmanSuccessor(actionLists[i])
            if tempFunState is None:
                return 0, False
        else:
            break
    return scoreEvaluation(tempFunState), True

"""
Function : sortFun(actionLists, score)

This function builds a dictionary by using sequence of actions and their relevant 
scores, sorts the dictionary and give each sequence rank according to their score.

Input : actionLists : list of sequences
        state : score of each sequence
        
Output : sortedList : Sorted Dictionary made from given input.
"""
def sortFun(actionLists, score):
    sortedList = []
    
    for i in range(0, 8):
        dictionary = {}
        dictionary["actions"] = actionLists[i]
        dictionary["score"] = score[i]
        dictionary["rank"] = 0
        sortedList.append(dictionary)
    sortedList.sort(key=lambda x:x["score"])
    for i in range(0,8):
        sortedList[i]["rank"] = i+1
    return sortedList

"""
Function : choiceWithProbability(sortedList)

This function returns the random tupple from a dictionary with its relative 
probability which is represented by its rank 

Input : sortedList : sorted dictionray

Output : item : Randomly chosen parent with its weighted probability
"""
def choiceWithProbability(sortedList):
    total_prob = sum(item["rank"] for item in sortedList)
    chosen = random.uniform(0, total_prob)
    cumulative = 0
    for item in sortedList:
        cumulative += item["rank"]
        if cumulative > chosen:
            return item

"""
Function : crossover(actions1, actions2)

This function is a crossover function that created child with 50% probability of
getting a gene from either of the parents.

Input : actions1 : sequence of actions of first parent
        actions2 : sequence of actions of second parent
        
Output : child1 : child1 created by crossover of two parents.
         child2 : child2 created by crossover of two parents.
"""
def crossover(actions1, actions2):
    child1 = []
    child2 = []
    for i in range(0,5):
        if random.random() > 0.5:
            child1.append(actions1[i])
        else:
            child1.append(actions2[i])
            
    for i in range(0,5):
        if random.random() > 0.5:
            child2.append(actions1[i])
        else:
            child2.append(actions2[i])
    return child1, child2

"""
Function : mutate(child, state)

This function mutates randomly chosen action of a sequence by random action.

Input : child : child node whose actions need to be mutated
        state : state where the original pacman is in the game.
"""
def mutate(child, state):
    possible = state.getAllPossibleActions()
    r1 = random.randint(0,4)
    child["actions"][r1] = possible[random.randint(0, len(possible)-1)]

"""
Function : copyActions(children)

This function copies the action sequences from children to parent.

Input : children : a dictionary that has all the information about children

Output : l1 : gives the sequence of actions of each child
"""
def copyActions(children):
    l1 = [[],[],[],[],[],[],[],[]]
    for i in range(0,8):
        tempList = children[i]["actions"]
        
        for j in range(0,5):
            l1[i].append(tempList[j])
    return l1    

"""
Function : copyScores(children)

This function copies scores of each child to a separate array.

Input : children : a dictionary that has all the information about children

Output : l1 : scores of each child
"""
def copyScores(children):
    l1 = [float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")]

    for i in range(0,8):
        l1[i] = children[i]["score"]

    return l1;

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [[],[],[],[],[],[],[],[]]
        self.childrenList = [[], []]
        self.scores = [float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")]
        for j in range(0,8):
            for i in range(0,5):
                self.actionList[j].append(Directions.STOP)
        for j in range(0,2):
            for i in range(0,5):
                self.childrenList[j].append(Directions.STOP)
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        #Initializes some variables.
        finalAction = Directions.STOP
        flag = True
        
        #assignes random action sequences to parents
        possible = state.getAllPossibleActions();
        for j in range(0,8):
            for i in range(0, len(self.actionList[j])):
                self.actionList[j][i] = possible[random.randint(0, len(possible)-1)]
        for i in range(0,8):
            self.scores[i], flag = evalFun(self.actionList[i], state)
        children = []
        
        #get children and make crosover and mutate them accordingly to find the best one.
        while flag:
            for i in range(0,4):
                sortedList = sortFun(self.actionList, self.scores)
                child1 = choiceWithProbability(sortedList)
                child2 = choiceWithProbability(sortedList)
                
                #do a crosover if the random test gives less than 0.7
                if random.random() <= 0.7:
                    child1["actions"], child2["actions"] = crossover(child1["actions"], child2["actions"])
                children.append(child1)
                children.append(child2)
            
            #mutate children if random test gives less than 0.1
            for i in range(0,8):
                if random.random() <= 0.1:
                    mutate(children[i], state)
                    
            #evaluate the scores of each children and assign them accordingly        
            for i in range(0,8):
                children[i]["score"], flag = evalFun(children[i]["actions"], state)
            
            #update the parents
            self.actionList = copyActions(children)
            self.scores = copyScores(children)
            if flag is True:
                del(children)
                children = []
        
        #Sort the children dictionary to get the best child        
        children.sort(key=lambda x:x["score"])
        finalActions = children[7]["actions"]
        finalAction = finalActions[0]
        
        #return the action[0] of best child
        return finalAction


    
###############################################################################
#
#   Genetic Algorithm starts here.
#
###############################################################################    
"""
Function : initNode(node)

This function initializes the child node with default values.

Input : node : parent node of the child

Output : Initializes the child node with default values
"""
def initNode(node):
    temp = {}
    temp["parent"] = node
    temp["children"] = False
    temp["visitCount"] = 0
    temp["score"] = 0
    temp["child_action_list"] = []
    temp["child_list"] = []
    
    return temp

"""
Function : backPropogation(node, score)

This function backpropogates the change in score and visit count till the root.

Input : node : current node for which the UCB score is just computed.
        score : computed UCB score
"""
def backPropogation(node,score):
    while node:
        node["score"] = node["score"] + score
        node["visitCount"] = node["visitCount"] + 1
        node = node["parent"]

"""
Function : select(node)

This function selects one child from the already expanded children by computing UCB scores

Input : node : parent node for which the child needs to be selected.

Output : bestChild : returns the child with the highest UCB score.
"""
def select(node):
    children = node["child_list"]
    maxScore = float("-inf")
    for child in children:
        totalScore = (normalizedScoreEvaluation(node["state"],child["state"])) + (math.sqrt((math.log(node["visitCount"])) / (child["visitCount"])))
        if totalScore > maxScore:
            maxScore = totalScore
            bestChild = child
    return bestChild



class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.flag = True
        return;

    """
    Function : expand(self, node)
    
    Input : node : parent node that needs to be expanded into child node
    
    Output : child : returns the expanded child
    """
    def expand(self, node):
        state = node["state"]
        legal = state.getLegalPacmanActions()
        #removes actions that are already expanded
        if node is not None:
            for i in range(0, len(node["child_action_list"])):
                action = node["child_action_list"][i]
                if action in legal:
                    legal.remove(action)
        
        #creates child by expanding random action from available actions
        if not node["children"]:
            currentAction = legal[random.randint(0,len(legal)-1)]
            instances = state.generatePacmanSuccessor(currentAction)
            if instances is None:
                self.flag = False
                return None
            child = {}
            child = initNode(node)
            child["action"] = currentAction
            child["state"] = instances
            node["child_list"].append(child)
        node["child_action_list"].append(currentAction)
        
        if len(legal) == 1:
            node["children"] = True
        return child


    """
    Function : tree(self, node)
    
    This function acts as default tree policy
    
    Input : node : root node
    
    Output : node : the expanded child node
    """
    def tree(self,node):
        while node["state"].isWin() + node["state"].isLose() == 0:
            if node["children"] == False:
                return self.expand(node)
            else:
                node = select(node)
                if node["state"] is None:
                    return None
        return node

    """
    Function : default(self, state)
    
    This function acts as default scoring policy
    
    Input : state : expanded node's state
    
    Output : scoreEvaluation() : absolute score of the expanded child
    """
    def score(self,state):
        rollout = 0
        while rollout < 5:
            if state.isWin() or state.isLose():
                return scoreEvaluation(state)
            else:
                legal = state.getLegalPacmanActions()
                if legal:
                    random_action = legal[random.randint(0, len(legal) - 1)]
                    state = state.generatePacmanSuccessor(random_action)
                    if state is None:
                        self.flag = False
                        return 0
            rollout = rollout + 1
        return scoreEvaluation(state)

    # GetAction Function: Called with every frame
    def getAction(self, state):        
        #initializes values
        node = {}
        node = initNode(None)
        node["action"] = None
        node["state"] = state

        #computes the MCTS
        while True:
            expanded_node = self.tree(node)
            if expanded_node is not None:
                score = self.score(expanded_node["state"])
                backPropogation(expanded_node,score)
            else:
                break

        #Computes the best child based on max visited nodes
        struct = []
        actions = node["child_action_list"]
        children = node["child_list"]
        for i in range(0, len(node["child_action_list"])):
            temp = {}
            temp["action"] = actions[i]
            tempList = children[i]
            temp["visitCount"] = tempList["visitCount"]
            struct.append(temp)
        struct.sort(key=lambda x: x["visitCount"])
        best = struct[len(struct) - 1]

        #returns the action that leads to best child
        return best["action"]
