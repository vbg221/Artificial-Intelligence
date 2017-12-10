import pandas as pd 
import numpy as np
import time
import random
import math
from sklearn.metrics import accuracy_score

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")



#dataset.head(6)
#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
    return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
    tr = int(len(dataset)*ratio)
    return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
    features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
    labels = dataset["winner"].astype(int).values
    return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
    f, l = getNumpy(dataset)
    return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
     predictions = np.array(solutions)
     labels = np.array(real)
     return (predictions == labels).sum() / float(labels.size)









data = dataset[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea', 'winner']]
train, test = trainingTestData(data, 0.7)

#label = train[['winner']]
#train = train[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']]

#target = test[['winner']]
#test = test[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']]











#==============================================================================
#==============================================================================
# # KNN Starts here
#==============================================================================
#==============================================================================
class KNN:
    def __init__(self, data):
        #KNN state here
        #Feel free to add methods
        self.k = 5
        self.dataset = data
        #print self.dataset

    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        #neightbour = pd.DataFrame()
        self.labels = labels

        self.dataset = encodeData(features, ['can_off', 'can_inc_cha_ope_sea'])
#        print self.dataset
        #categories = None
        self.dataset = normalizeData(self.dataset, ['net_ope_exp', 'net_con', 'tot_loa'])
#        print self.dataset

#        print "After adding distance : \n"
#        print self.dataset

    def countMax(self, l):
        true = 0
        false = 0
        #print l
        for i in range(0,self.k):
            #print l[i]
            if(l[i]['winner']):
                true += 1
                #print "True"
            else:
                false += 1
                #print "False"
        if( true > false):
            return True
        else:
            return False

    def calculateDistance(self, record):
        output = None
        #categories = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off_H', 'can_off_P', 'can_off_S', 'can_inc_cha_ope_sea_CHALLENGER', 'can_inc_cha_ope_sea_INCUMBENT', 'can_inc_cha_ope_sea_OPEN']
        categories = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        l = []
        #flag = False
#        print categories
#        print "\n"
        for i, tupple in enumerate(self.dataset.values):#range(0, len(self.dataset)):#tupple in self.dataset.itertuples():
            #tupple = self.dataset.iloc[i]


#            if(flag is False):
#                flag = True
#                continue
            
            distance = 0
            node = {}
            for cat in categories:
#                print cat
#                print "Record"
#                print record
#                print "\n"
#                print "Tupple"
#                print tupple
#                print "\n"
#                print record['net_ope_exp']
#                print tupple[cat]
                distance += ((record[cat] - tupple[cat])**2)
            node['distance'] = np.sqrt(distance)
            node['winner'] = self.labels.iloc[i]['winner']#tupple['winner']
            l.append(node)
            
        l.sort(key=lambda x:x['distance'])
        #print l
        output = self.countMax(l)
#        print output
        return output

    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        test = pd.DataFrame(encodeData(features, ['can_off', 'can_inc_cha_ope_sea']))
        test = pd.DataFrame(normalizeData(test, ['net_ope_exp', 'net_con', 'tot_loa']))
        #flag = False
        #print test
        label = []
        for i, record in enumerate(test.values):#range(0, len(test)):#record in test.itertuples():

#            if flag is False:
#                print record
#                flag = True
#                continue
#            print "Hihahahahaha"
            #record = test.iloc[i]
            label.append(self.calculateDistance(record))

        return label

    


#==============================================================================
# 
# def accuracy(labels, target):
#     true = 0
#     false = 0
#     t = []
#     for val in target['winner']:
#         t.append(val)
#     #print t
#     for i in range(0, len(labels)):
#         if (labels[i] == t[i]):
#             true += 1
#         else:
#             false += 1
#     score = (float)(true/len(labels))
# #        if len(labels) == len(target):
# #            score = True
# #        else:
# #            score = False
#     return score



#==============================================================================
#==============================================================================
#==============================================================================
# # Code to use KNN algorithm
#==============================================================================
#==============================================================================

"""
st = time.time()
#data = pd.DataFrame(dataset, columns = {'net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea'})
data = dataset[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea', 'winner']]
train, test = trainingTestData(data, 0.7)

label = train[['winner']]
train = train[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']]

target = test[['winner']]
test = test[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']]

df = KNN(train)
df.train(train, label)

labs = df.predict(test)

#print "labels : ", labs
#print "target : ", target
#print accuracy(labs, target['winner'])
et = time.time()
print "Elapsed time : ", (et - st)
#print evaluate(labs, target['winner'])

print accuracy_score(target['winner'], labs)

"""
#==============================================================================
#==============================================================================
# # Perceptron Starts here
#==============================================================================
#==============================================================================

"""
X ubho
b ubho
weight square
"""

class Perceptron:
    def __init__(self):
        #Perceptron state here
        #Feel free to add methods
        self.weights = np.asmatrix(np.full((9,1), random.random()))
        #print self.weights
        self.bias = random.random()#np.asmatrix(np.full((1,1), random.random()))
        #print self.bias
        self.alpha = 0.01
        #print "Initialization complete."
        #print "weights and bias are : ", self.weights, self.bias

    def evaluate(self, x):
        out = (x * self.weights) + self.bias
        #print "Multiplication : ", out#(np.asmatrix(self.weights)* np.asmatrix(x))
        
        return out
    def update(self, x, out, target):
        #print "out : ", out
        #print "target : ", target
        if target:
            t = 1
        else:
            t = 0
        #print "t : ", t
        #print "\nBefore Update : "
        #print "\nWeights : ", self.weights, self.bias
        #print out[0][0]
        #print x.T
        self.weights = self.weights + ((self.alpha*(t - np.asscalar(out)))*x.T)
        self.bias = self.bias + (self.alpha*(t - out))
        #a = (self.alpha*(target - out))
        #return
        #print "\nAfter Update : "
        #print "\nWeights : ", self.weights, self.bias
        
    def limitevaluate(self, x):
        ans = self.evaluate(x)
        #print ans
        if ans >= 0.5:
            return 1
        elif ans < 0.5:
            return 0


    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels

        #print "In train"

        """
        One hot encode data and then normalize it.
        """
        feature = pd.DataFrame(encodeData(features, ['can_off', 'can_inc_cha_ope_sea']))
        feature = pd.DataFrame(normalizeData(feature, ['net_ope_exp', 'net_con', 'tot_loa']))


        for i, tup in enumerate(feature.values):
            #print "Tupple : ", tup
            out = self.evaluate(np.asmatrix(tup))
            self.update(np.asmatrix(tup), out, labels.iloc[i])
        #return
        #print "training complete."
        #print "weights are : ", self.weights, self.bias


    def predict(self, features):
        #print "Predicting ... ... ..."
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        features = pd.DataFrame(encodeData(features, ['can_off', 'can_inc_cha_ope_sea']))
        features = pd.DataFrame(normalizeData(features, ['net_ope_exp', 'net_con', 'tot_loa']))

        output = []
        for index, row in enumerate(features.values):
            output.append(self.limitevaluate(np.asmatrix(row)))

        #print "Completed prediction. Returning to main program... ..."
        return output

"""
percept = Perceptron()
percept.train(train, label['winner'])
output = percept.predict(test)

print accuracy_score(target['winner'], output)
"""

class MLP:
    def __init__(self):
        #Multilayer perceptron state here
        #Feel free to add methods
        
        """
        c = np.asmatrix(np.random.rand(3, 3))
        """
        
        self.layers = 2
        self.weights = []
        self.bias = []

        self.numberOfNeurons = []
        self.numberOfNeurons.append(9)
        self.numberOfNeurons.append(9)
        self.numberOfNeurons.append(1)

        self.alpha = 0.01
        
        self.neurons = []
        self.neurons.append(np.asmatrix(np.full((1,9), 0.00000000000)))
        self.neurons.append(np.asmatrix(np.full((1,9), 0.00000000000)))
        self.neurons.append(np.asmatrix(np.full((1,1), 0.00000000000)))
        
        self.s = []
        self.s.append(np.asmatrix(np.full((9,1), 0.00000000000)))
        self.s.append(np.asmatrix(np.full((1,1), 0.00000000000)))
        
        self.n = []
        self.n.append(np.asmatrix(np.full((1,9), 0)))
        self.n.append(np.asmatrix(np.full((1,1), 0)))
        
        self.weights.append(np.asmatrix(np.random.rand(9, 9))-0.5)
        self.bias.append(np.asmatrix(np.random.rand(1, 9))-0.5)
        
        self.weights.append(np.asmatrix(np.random.rand(9, 1))-0.5)
        self.bias.append(np.asmatrix(np.random.rand(1, 1))-0.5)
        
        print "Weights : "
        for weight in self.weights:
            print np.shape(weight)
        print "Bias : "
        for bias in self.bias:
            print np.shape(bias)
        print "Neurons : "
        for neuron in self.neurons:
            print np.shape(neuron)
        print ""
        
    def preprocessMLP(self, classDataset):
        
        data = classDataset[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea', 'winner']]
        #train, test = trainingTestData(data, 0.7)
        
        labels = np.asarray(data[['winner']])
        temp = data[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']]
        
        #target = test[['winner']]
        #test = test[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']]
        
        features = pd.DataFrame(encodeData(temp, ['can_off', 'can_inc_cha_ope_sea']))
        features = pd.DataFrame(normalizeData(features, ['net_ope_exp', 'net_con', 'tot_loa']))
        
        return features, labels

    def transferFun(self, x):
        return (1/(1 + np.exp(-x)))
    
    def defTransferFun(self, x):
        return (self.transferFun(x) * (1 - self.transferFun(x)))

    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        st = time.time()
        while (time.time() - st) < 25.000 :
            for index, record in enumerate(features.values):
                #print "\n\n\n"
                self.neurons[0] = np.asmatrix(record)
                for i in range(0, self.layers):
    #                print np.shape(self.neurons[i])
    #                print np.shape(self.weights[i])
    #                print np.shape(self.bias[i])
    #                print "Shape of Matrix Arithmatic : "
    #                print np.shape((np.asmatrix(self.neurons[i]) * np.asmatrix(self.weights[i])) + np.asmatrix(self.bias[i]))
    #                print ""
                    self.n[i] = (self.neurons[i] * self.weights[i]) + self.bias[i]
                    self.neurons[i+1] = self.transferFun(self.n[i])
                
                
                """
                The code segment written below is hardcoded for networks with 1 final output neuron only.
                For networks with different dimensions, below written code segment must be executed for all individual final output neuron.
                """
                #print labels[index]
                if labels[index]:
                    l = 1
                else:
                    l = 0
                #print "l : ", l
                #print "neurons : ", self.neurons[2]
                self.s[1] = ((self.neurons[2] - l)) * self.defTransferFun(self.n[1])
                #print "Sensitivity : ", self.s[1]
                """
                Hardcoded segment ends here.
                """
                
                
                for j in range (0, self.numberOfNeurons[1]):
                    #print "S : ", np.shape(self.s[0][j])
                    #print "Output of defTransferFun : "
                    #print "transferFun input : ", np.shape(self.n[0])
                    #print "n[0][0, j] : ", self.n[0][0, j]
                    #a = self.n[0]
                    #print a
                    #print a[0][0]
                    total = 0
                    for element in self.weights[1][j]:
                        total += (element * self.s[1])
                    y = self.defTransferFun(self.n[0][0, j]) * total
                    self.s[0][j, 0] = y[0, 0]
                    
                """
                point e from the slides of professor Kok Ming Leung remains unimplemented.
                You will need to put the whole function through loop after the above mentioned point is dealt with.
                """
                
                for i in range(0, self.layers):
                    #print "Iteration : ", i
                    #print "neurons : ", np.shape(self.neurons[i])
                    #print "s : ", np.shape(self.s[i])
                    self.weights[i] = self.weights[i] - (self.alpha * (self.neurons[i].T * self.s[i].T))
                    self.bias[i] = self.bias[i] - (self.alpha * self.s[i].T)
                
                
#        print "\nWeights : "
#        for weight in self.weights:
#            print np.shape(weight)
#        print "Bias : "
#        for bias in self.bias:
#            print np.shape(bias)
#        print "Neurons : "
#        for neuron in self.neurons:
#            print np.shape(neuron)
    #                
    #            
    #            print "Weights : "
    #            for weight in self.weights:
    #                print weight
    #            print "Bias : "
    #            for bias in self.bias:
    #                print bias
    #            print "Neurons : "
    #            for neuron in self.neurons:
    #                print neuron
                #return

    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        output = []
        for index, record in enumerate(features.values):
            out = []
            self.neurons[0] = np.asmatrix(record)
            for i in range(0, self.layers):
                self.n[i] = (self.neurons[i] * self.weights[i]) + self.bias[i]
                self.neurons[i+1] = self.transferFun(self.n[i])
            if self.neurons[2][0, 0] > 0.5:
                out.append(True)
            else:
                out.append(False)
            output.append(out)
            
            #output.append(self.neurons[2][0, 0])
        return np.asarray(output)
        
"""
mlp = MLP()
features, labels = mlp.preprocessMLP(train)
st = time.time()
mlp.train(features, labels)
et = time.time()
print "Training time : ", (et - st)
test_features, target = mlp.preprocessMLP(test)
predicted = mlp.predict(test_features)
print accuracy_score(target, predicted)
print "\n", evaluate(predicted, target)
"""



class ID3:
    def __init__(self):
        #Decision tree state here
        #Feel free to add methods
        self.node = []

    def categorize(self, temp):
        category = ['net_ope_exp', 'net_con', 'tot_loa']
        for cat in category:
            mini = temp[cat].min()
            maxi = temp[cat].max()
            diff = (maxi - mini) / 5.00000000000
            for index, row in temp.iterrows():
                if (row[cat] > mini) and (row[cat] < (mini + diff)):
                    temp.set_value(index, cat, 0)
                elif (row[cat] > (mini + (diff))) and (row[cat] < (mini + (2 * diff))):
                    temp.set_value(index, cat, 1)
                elif (row[cat] > (mini + (2 * diff))) and (row[cat] < (mini + (3 * diff))):
                    temp.set_value(index, cat, 2)
                elif (row[cat] > (mini + (3 * diff))) and (row[cat] < (mini + (4 * diff))):
                    temp.set_value(index, cat, 3)
                else:
                    temp.set_value(index, cat, 4)
        return temp

    def preprocessID3(self, classDataset):
        data = classDataset[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea', 'winner']]
        
        labels = np.asarray(data[['winner']])
        temp = data[['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']]
        
        features = self.categorize(temp)
        print features
        return features, labels

    def buildTree(self,node, cat, uniqueue_val):
        temp = {}
        temp['cat'] = cat
        temp['children'] = []
        temp['val'] = uniqueue_val
        node.append(temp)
        return temp['children']

    def calculateEntropy(self, node, features, labels):
        entropy = []
        """
        These categories will change dynamically later on
        """
        category = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
        
        for cat in category:
            total = {}
            total['cat'] = cat
            listOfUniqueValues = features[cat].unique()
            total['unique'] = len(listOfUniqueValues)
            total['uniqueue_val'] = listOfUniqueValues
            uniques = []
            t = 0
            labels = np.asarray(labels)
            pos = (labels == True).sum()/float(labels.size)
            neg = (labels == False).sum()/float(labels.size)
            datasetEntropy = (pos * math.log(pos, 2)) + (neg * math.log(neg, 2))
            total['gain'] = datasetEntropy
            for val in listOfUniqueValues:
                temp = {}
                temp['val'] = val
                temp['true'] = 0
                temp['false'] = 0
                temp['total'] = 0
                temp['entropy'] = 0
                uniques.append(temp)
            for index, row in features.iterrows():
                for i, dic in enumerate(uniques):
                    if dic['val'] == row[cat]:
                        if labels[index]:
                            dic['true'] += 1
                            dic['total'] += 1
                            t += 1
                        else:
                            dic['false'] += 1
                            dic['total'] += 1
                            t += 1
            for val in enumerate(uniques):
                if val[1]['true'] is not 0 and val[1]['false'] is not 0:
                    val[1]['entropy'] = (((val[1]['true']/float(val[1]['total'])) * math.log((val[1]['true']/float(val[1]['total'])), 2)) + (((val[1]['false'])/float(val[1]['false'])) * math.log(((val[1]['false'])/float(val[1]['total'])), 2)))
                    total['gain'] -= ((val[1]['total']/float(t)) * val[1]['entropy'])
            entropy.append(total)
            entropy.sort(key = lambda x:x['gain'])
        print entropy
        #print entropy.pop()
        final_record = entropy.pop()
        node = self.buildTree(node, final_record['cat'], final_record['uniqueue_val'])
        f = []
        for val in final_record['uniqueue_val']:
            dta = pd.DataFrame(features.loc[features[final_record['cat']] == val])
            #dta.
            dta.drop(final_record['cat'], inplace = True, axis = 1)
            f.append(dta)
        l = []
        for df in f:
            la = []
            
            print df.axes[0]
            
            for index in enumerate(df.axes[0]):
                la.append(labels[index[1]])
                #print index[1]
            l.append(la)
            
        for fa,la in zip(f, l):
            print "F : ", fa
            print "L : ", la
            print "Done"
        
        return node, features, labels

    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        self.features = features
        self.labels = labels
        node = self.node
        node, features, labels = self.calculateEntropy(node, features, labels)
        

    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        print "Come on"

id3 = ID3()
features, labels = id3.preprocessID3(train)
id3.train(features, labels)
