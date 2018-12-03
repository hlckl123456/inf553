Include /PA in this directory if you wish to run the script

File: randomWalk.py
This Python script create a networkX tripartite graph with the PA dataset if there isn't one.
You would need to move the PA dataset in this folder in order for this to run.

The script then performs random walk on the generated graph with 5000 walk steps per node and a restart
alpha value of 0.05

To run this script call $python randomWalk.py T 
arguments:
T: run this on the test set
V: run this on the validation set

The result is a pickle saved randomWalkResult_test.txt file + random_walk_hit_ratio.txt file