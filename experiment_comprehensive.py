# -*- coding: utf-8 -*-
import sys
sys.path.append("D:/Anaconda/files/MDP/secret")
import file_mdp
import random
random.seed(0)
import matplotlib.pyplot as plt
from model_free import *


if __name__ == "__main__":
    read_best()
    plt.figure(figsize=(12,6))


    mc(num_iter1 = 1000, epsilon = 0.1)
    mc(num_iter1 = 1000, epsilon = 0.2)
    sarsa(num_iter1 = 1000, alpha = 0.2,  epsilon = 0.1);
    sarsa(num_iter1 = 1000, alpha = 0.4,  epsilon = 0.1);
    sarsa(num_iter1 = 1000, alpha = 0.2,  epsilon = 0.2);
    sarsa(num_iter1 = 1000, alpha = 0.4,  epsilon = 0.2); 
    qlearning(num_iter1 = 1000, alpha = 0.2,  epsilon = 0.1);
    qlearning(num_iter1 = 1000, alpha = 0.4,  epsilon = 0.1);
    qlearning(num_iter1 = 1000, alpha = 0.2,  epsilon = 0.2);
    qlearning(num_iter1 = 1000, alpha = 0.4,  epsilon = 0.2);


    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend()
    plt.show();
#对比三种算法在机器人找金币找个场景下 Qlearning性能最好
#其次SARSA 最后是MC Control