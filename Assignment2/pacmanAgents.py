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

def climb(actionLists, state, flag):
    tempFunState = state
    for i in range(0,len(actionLists)):
        if tempFunState.isWin() + tempFunState.isLose() == 0:
            tempFunState = tempFunState.generatePacmanSuccessor(actionLists[i])
        else:
            flag = False
            break
    if (tempFunState is not None):
        score = scoreEvaluation(tempFunState) 
    elif tempFunState.isWin():
        score = 3000
    elif tempFunState.isLose():
        score = -3000
    else:
        score = 0
    return score, flag


class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    
    
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,5):
            self.actionList.append(Directions.STOP)       
        return;


    # GetAction Function: Called with every frame
    def getAction(self, state):
        finalScore = -500 #float("-inf") #
        currentScore = float("-inf")
        finalAction = Directions.STOP
        flag = True
        
        possible = state.getAllPossibleActions()
        for i in range(0, len(self.actionList)):
            self.actionList[i] = possible[random.randint(0, len(possible)-1)]

        currentScore, flag = climb(self.actionList, state, flag)
        finalScore = currentScore
        finalAction = self.actionList[0]

        while(True):
            # get all legal actions for pacman

            #for i in range(0,len(self.actionList)):
            #    self.actionList[i] = possible[random.randint(0,len(possible)-1)];
            for i in range(0, len(self.actionList)):
                if random.randint(0,1) == 1:
                    self.actionList[i] = possible[random.randint(0, len(possible)-1)]

            print("Current Actionlist : ", self.actionList)


            currentScore, flag = climb(self.actionList, state, flag)


            print"Current Action of actionlist : ", self.actionList[0]
            print"Score at the end of current sequence : ", currentScore
            print
            
            if(finalScore <= currentScore):
                finalScore = currentScore
                finalAction = self.actionList[0]


            print"Final action till now from all iterations : ", finalAction
            print"Maximum score till now : ", finalScore
            print"--------------------------------------------------"            


            if (finalScore > currentScore) or flag == False :
                break
        
        legalActions = state.getLegalPacmanActions()
        if finalAction not in legalActions:
            finalAction = Directions.STOP
        
        print"#######################################"
        print"Chosen Action : ", finalAction
        print"Score for chosen action : ", finalScore
        print"#######################################"
        
        
        return finalAction;
    
    

def evalFun(actionLists, state):
    tempFunState = state
    for i in range(0,len(actionLists)):
        if tempFunState.isWin() + tempFunState.isLose() == 0:
            tempFunState = tempFunState.generatePacmanSuccessor(actionLists[i])
        else:
            break
    return scoreEvaluation(tempFunState)


def sortFun(actionLists, score):
    sortedList = []
    
    for i in range(0, 8):
        dictionary = {}
        dictionary["actions"] = actionLists[i]
        dictionary["score"] = score[i]
        dictionary["rank"] = 0
        sortedList.append(dictionary)
    sortedList.sort(key=lambda x:x["score"])
    #print sortedList
    for i in range(0,8):
        sortedList[i]["rank"] = i+1
    #print sortedList
    print"---------------------------"
    return sortedList

#from random import uniform
def choiceWithProbability(sortedList):
    total_prob = sum(item["rank"] for item in sortedList)
    chosen = random.uniform(0, total_prob)
    cumulative = 0
    for item in sortedList:
        cumulative += item["rank"]
        if cumulative > chosen:
            return item

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

def mutate(child, state):
    possible = state.getAllPossibleActions()
    r1 = random.randint(0,4)
    print"r1 : ",r1
    child["actions"][r1] = possible[random.randint(0, len(possible)-1)]

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [[],[],[],[],[],[],[],[]];
        self.childrenList = [[], []]
        self.scores = [float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")]
        for j in range(0,8):
            for i in range(0,5):
                self.actionList[j].append(Directions.STOP)
        for j in range(0,2):
            for i in range(0,5):
                self.childrenList[j].append(Directions.STOP)
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        finalAction = Directions.STOP
        
        possible = state.getAllPossibleActions();
        for j in range(0,8):
            for i in range(0, len(self.actionList[j])):
                self.actionList[j][i] = possible[random.randint(0, len(possible)-1)]
        #print self.actionList
        for i in range(0,8):
            self.scores[i] = evalFun(self.actionList[i], state)
        #print self.scores
        
        
        children = []
        
        for i in range(0,4):
            sortedList = sortFun(self.actionList, self.scores)
            child1 = choiceWithProbability(sortedList)
            child2 = choiceWithProbability(sortedList)
            if random.random() <= 0.7:
                child1["actions"], child2["actions"] = crossover(child1["actions"], child2["actions"])
                   
            #print"###########################################"
            #print child1
            #print child2
            #print"###########################################"

            children.append(child1)
            children.append(child2)
         
        for i in range(0,8):
            if random.random() <= 0.1:
                mutate(children[i], state)
                
                
        for i in range(0,8):
            children[i]["score"] = evalFun(children[i]["actions"], state)
            
        children.sort(key=lambda x:x["score"])
        finalActions = children[7]["actions"]
        finalAction = finalActions[0]
        #print children

        print finalAction
        print"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"    
        return finalAction

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        return Directions.STOP
