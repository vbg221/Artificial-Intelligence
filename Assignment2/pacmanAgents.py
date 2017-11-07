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
            if tempFunState is None:
                flag = False
                return 0, flag
        else:
            flag = False
            break
    
    """
    if (tempFunState is not None):
        score = scoreEvaluation(tempFunState) 
    elif tempFunState.isWin():
        score = 3000
    elif tempFunState.isLose():
        score = -3000
    else:
        score = 0
    """
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

            #print("Current Actionlist : ", self.actionList)


            currentScore, flag = climb(self.actionList, state, flag)


            #print"Current Action of actionlist : ", self.actionList[0]
            #print"Score at the end of current sequence : ", currentScore
            #print
            
            if(finalScore <= currentScore):
                finalScore = currentScore
                finalAction = self.actionList[0]


            #print"Final action till now from all iterations : ", finalAction
            #print"Maximum score till now : ", finalScore
            #print"--------------------------------------------------"            


            if flag == False :
                break
        
        legalActions = state.getLegalPacmanActions()
        if finalAction not in legalActions:
            finalAction = Directions.STOP
        
        #print"#######################################"
        #print"Chosen Action : ", finalAction
        #print"Score for chosen action : ", finalScore
        #print"#######################################"
        
        
        return finalAction;
    
    

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
    #print"---------------------------"
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
    #print"r1 : ",r1
    child["actions"][r1] = possible[random.randint(0, len(possible)-1)]

def copyActions(children):
    l1 = [[],[],[],[],[],[],[],[]]
    #print children
    for i in range(0,8):
        tempList = children[i]["actions"]
        
        for j in range(0,5):
            l1[i].append(tempList[j])
    
    print"After swapping : ",l1
    return l1    

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
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        finalAction = Directions.STOP
        flag = True
        possible = state.getAllPossibleActions();
        for j in range(0,8):
            for i in range(0, len(self.actionList[j])):
                self.actionList[j][i] = possible[random.randint(0, len(possible)-1)]
        #print self.actionList
        for i in range(0,8):
            self.scores[i], flag = evalFun(self.actionList[i], state)
        #print self.scores
        
        
        children = []
        while flag:
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
                children[i]["score"], flag = evalFun(children[i]["actions"], state)
                
            #print self.actionList
            #print self.scores
            #print "------------------------------------------------------------"
            print children
            print "------------------------------------------------------------"
            self.actionList = copyActions(children)
            self.scores = copyScores(children)
            #print self.actionList
            #print self.scores
            #print "############################################################"
            if flag is True:
                del(children)
                children = []
                
        children.sort(key=lambda x:x["score"])
        finalActions = children[7]["actions"]
        finalAction = finalActions[0]
        #print children

        #print finalAction
        print"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"    
        return finalAction
"""
class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        return Directions.STOP
"""

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    def getUCBScore(self, node, parent_state):
        exploitation = normalizedScoreEvaluation(parent_state, node["state"])
        c = 1
        parent_node = node["parent"]
        if node["number_of_visits"] == 0:
            node["number_of_visits"] = 0.001
        exploration = c * math.sqrt((math.log(parent_node["number_of_visits"])) / (node["number_of_visits"]))
        UCB_Score = exploitation + exploration
        #print UCB_Score
        return UCB_Score

    def DefaultPolicy(self, new_state):
        for i in range(0, 10):
            
            if not new_state.isWin() + new_state.isLose() == 0:
                return scoreEvaluation(new_state)
            
            else:
                legal_actions = new_state.getLegalPacmanActions()
                random_action = legal_actions[random.randint(0, len(legal_actions) - 1)];
                new_state = new_state.generatePacmanSuccessor(random_action)
        return scoreEvaluation(new_state)

    expanded_list = []

    def Expand(self, node):
        state = node["state"]
        legal_actions = state.getLegalPacmanActions()
        # successors = [(state.generatePacmanSuccessor(action), action) for action in legal_actions]
        current_action = None
        #print legal_actions
        if node is not None:
            for action in node["child_action_list"]:
                if action in legal_actions:
                    legal_actions.remove(action)
            
        if not node["children"]:
            #current_action = legal_actions[0]
            current_action = legal_actions[random.randint(0,len(legal_actions)-1)]
            
            
            #print "random_action"
            #print current_action
            new_state = state.generatePacmanSuccessor(current_action)
            new_node = {}
            new_node["parent"] = node
            new_node["state"] = new_state
            new_node["action"] = current_action
            new_node["number_of_visits"] = 0
            new_node["children"] = False
            new_node["UCB_score"] = 999999
            new_node["child_action_list"] = []
            new_node["child_list"] = []

            node["child_list"].append(new_node)
            self.expanded_list.append(new_node)

        node["child_action_list"].append(current_action)
        if len(legal_actions) == 1:
                node["children"] = True

            

        return new_node

    def Select(self, node, c, root_state):
        legal_actions = node["child_action_list"]
        #print"befor loop in select"
        for child_node in node["child_list"]:
            child_node["UCB_score"] = self.getUCBScore(child_node, root_state)
        
        #print"after loop in select"
        max = -999999
        for child in node["child_list"]:
            if child["UCB_score"] > max:
                max = child["UCB_score"]
                best_child = child
                #print max

        return best_child


    def TreePolicy(self, node, root_state):
        while node["state"].isWin() + node["state"].isLose() == 0:
            if node["children"] == False:
                #print"Before Expand"
                return self.Expand(node)
            else:
                #print"before select function."
                node = self.Select(node, 1, root_state)
                #print"after select function."
        return node


    def Backup(self, node, delta):
        while node is not None:
            node["number_of_visits"] += 1
            #node["total_score"] += 1
            node = node["parent"]

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        root_state = state
        root_node = {}
        root_node["parent"] = None
        root_node["action"] = None
        root_node["state"] = root_state
        #root_node["total_score"] = 0
        root_node["number_of_visits"] = 0
        root_node["children"] = False
        root_node["child_list"] = []
        root_node["child_action_list"] = []
        root_node["UCB_score"] = 999999


        count_rollOuts = 0

        node = root_node
        state = root_state

        while(count_rollOuts < 5):
            new_node = self.TreePolicy(node, state)
            #print"First Line executed."
            delta = self.DefaultPolicy(new_node["state"])
            #print"Second line executed."
            count_rollOuts += 1
            #print"rollouts executed."
            self.Backup(new_node, delta)
            #print"back propogation executed."
            #print"rollout counts",count_rollOuts

        
        dictionary1=[]
        action_list = node["child_action_list"]
        child_list = node["child_list"]
        for i in range(0, len(node["child_action_list"])):
            #print i
            temp = {}
            temp["action"] = action_list[i]
            t1 = child_list[i]
            temp["visit_count"] = t1["number_of_visits"]
            dictionary1.append(temp)
        
        dictionary1.sort(key=lambda x:x["visit_count"])
        best = dictionary1[len(dictionary1)-1]
        #print"##################################"
        return best["action"]#Directions.STOP#action