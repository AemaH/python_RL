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
    mc(num_iter1 = 6000, epsilon = 0.2);
    mc(num_iter1 = 6000, epsilon = 0.2);
    mc(num_iter1 = 6000, epsilon = 0.2);
    mc(num_iter1 = 6000, epsilon = 0.2);
    sarsa(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);
    sarsa(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);
    sarsa(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);
    sarsa(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);
    qlearning(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);
    qlearning(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);
    qlearning(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);
    qlearning(num_iter1 = 6000, alpha = 0.2,  epsilon = 0.2);


    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend()
    plt.show();
#没什么好说的，就是将几种方法都试验一下,
#采用不同的迭代不同的epsilon和alpha试验一下情况
#比对参数选用的是当前策略 状态-动作价值
#和最优策略的状态-动作价值之间的平方差
#主要内容在model_free里面 毕竟这只是个画图
#主要目的也还是为了描述 判断这三种方法大致趋势
