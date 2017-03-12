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


    ############# variance ##################
    mc(num_iter1 = 5000, epsilon = 0.2);
    mc(num_iter1 = 5000, epsilon = 0.4);
    mc(num_iter1 = 5000, epsilon = 1.0); 

    sarsa(num_iter1 = 5000, alpha = 0.2,  epsilon = 0.2);
    sarsa(num_iter1 = 5000, alpha = 0.2,  epsilon = 0.4); 
    sarsa(num_iter1 = 5000, alpha = 0.2,  epsilon = 1.0)

        
    qlearning(num_iter1 = 5000, alpha = 0.2,  epsilon = 0.2);
    qlearning(num_iter1 = 5000, alpha = 0.2,  epsilon = 0.4);
    qlearning(num_iter1 = 5000, alpha = 0.2,  epsilon = 1.0);
    

    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend()
    plt.show();
#选取相同的迭代次数其他参数保持一样，差别只在epsilon大小的选取上
#参照图中了解MC Control和SARSA的epsilon选取对图像影响很大
#相反 Qlearning的epsilon变化对其几乎毫无影响，参照定义式中我们也可以知道
#毕竟epsilon影响的是动作选取，而这里的纵轴的误差是q和最优q的平方差