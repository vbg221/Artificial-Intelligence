# Artificial-Intelligence
This project is consist of multiple AI algorithms that are used by the PACMAN gaming agent to decide where to move next at any given second. In simple words, my algorithms provide direction to our little Pacman and try to show him the way to win the game.

Note: I wrote the algorithms in Python (2.7.13) in the Pacman framework provided by University of California, Berkeley for educational purposes. This project was a part of acedemic exercise for my course Artificial Intelligence at New York University.

## List of Algorithms used:
I have used various Artificial Intelligence algorithms. Here is the full list of all the algorithms. 
1. Depth First Search
2. Bredth First Search
3. A star
4. Hill Climber
5. Genetic Algorithm
6. Monte Carlo Tree Search

Apart from the algorithm used for Pacman, I implemented few more algorithms on a different dataset to predict and classify data into multiple classes.
The algorithms implemented for classification are:
1. K nearest neighbor
2. Perceptron
3. Multi Layer Perceptron
4. ID3 (Decision Tree)

## Usage Instructions:
This program runs locally and requires python 2.7 (preferably 2.7.13). The algorithms I wrote are in the PacmanAgents.py file. Rest of the files are framework that provides and supports the game.

To run the game, 
1. Clone the repository to your local machine. 
2. Run pacman.py from command line using
```
python pacman.py -p AgentName
```
AgentName can be replaced by specific agent i.e. DFSAgent, AStarAgent etc.

## Observations:
From the observations made by implementing these algorithms, I came to conclusion as below:
1. DFS and BFS (limited depth controlled by restricting time taken for each move) perform poorly in comparison to all the algorithms.
2. A star and Hill Climber gives fair accuracy by collecting more than half of the dots on the grid.
3. Even with the high time complexity and complex logic, Genetic algorithm and MCTS almost finishes the game (with occasional wins).

4. The second portion which consists of classification was tested by TAs on a separate dataset which was kept private and hence I have no merit to make any comparisons for those four algorithms.
But I did get fair training and cross validation accuracy of around 60-70 % on all algorithms implemented from scratch.
