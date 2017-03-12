#!/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("./secret")

import grid_mdp
import random
random.seed(0)
import numpy as np


class Policy:
    def __init__(self, grid, epsilon):
        self.actions = grid.actions
        #确定动作集合

        grid.start();
        #确定初始状态
        t,hats,r = grid.receive(self.actions[0]);
        #可以理解为下一句确定hats长度所用的 话说你用action[.]里面的数字是什么貌似都没影响吧
        self.theta = [ 0.0  for i in xrange(len(hats)*len(self.actions)) ]
        #hats的长度也就是feas，也就是参数theta的个数，每个状态s用特征表示有feas个，然后还有对应多个动作就是那么多个
        self.theta = np.array(self.theta);
        #建立矩阵
        self.theta = np.transpose(self.theta);
        #矩阵转置
        self.epsilon = epsilon

    def get_fea_vec(self, fea, a):
        f = np.array([0.0 for i in xrange(len(self.theta))]);
           #f是总共的特征个数（状态特征乘上动作） 
        idx = 0
        for i in xrange(len(self.actions)):
            if a == self.actions[i]: idx = i; 
            #让输入的动作a和actions集合中相应动作对应，然后让idx等于对应的动作标号   
        for i in xrange(len(fea)):
            f[i + idx * len(fea)] = fea[i];
            #比如一个fea也就是一个状态包含特征数为10，总共有5个动作，于是f包含50个；
            #然后第二个动作(从0开始 0 1 2 3 4)也就是 依次令f[10]=fea[0] f[11]=fea[1]...
        #实质上，这个函数的意义也就是在于将fea包含的状态特征放在关于动作a的部分
        #（毕竟用f一个矩阵来表示之前的状态s和相应动作a，表示方法也就是将总长分为|A|份，在第a份处放上状态s的特征，代表的也就是在这个状态s选用动作a）
        return f

    def qfunc(self, fea, a):
    #毕竟fea代表状态s，然后a就是动作
    #调用get_fea_vec()函数将此时的f表达出来
        f = self.get_fea_vec(fea, a);
        return np.dot(f, self.theta);
        #然后计算这个f和theta两者点积求出数值也就是q
        #毕竟动作和状态什么的都是借助矩阵表示描述出来了
        #那么计算q也就是靠这个参数theta
        #更新计算theta的方法也有update()函数描述出来


    def epsilon_greedy(self, fea):
        ## max q action
        epsilon = self.epsilon;

        amax    = 0
        qmax    = self.qfunc(fea, self.actions[0]) 
        for i in xrange(len(self.actions)):
            a   = self.actions[i]
            q   = self.qfunc(fea, a)
            if qmax < q:
                qmax  = q;
                amax  = i; 
            
        ##probability
        pro = [0.0 for i in xrange(len(self.actions))]
        pro[amax] += 1- epsilon
        for i in xrange(len(self.actions)):
            pro[i] += epsilon / len(self.actions)
    #看了下 ε-greedy的选择策略，最优的时候 也就是说让qfunc最大的时候pro计算概率是上面的加上下面的部分也就是
    #最优时候的概率，其他只包含后一部分表示各个动作a选取的概率
        ##choose
        r = random.random()
        s = 0.0
        for i in xrange(len(self.actions)):
            s += pro[i]
            #然后让我们看下r代表的意义，它表示一个非0数
            #再看下我们对各个动作概率做的事情：让他们累加
            #如果没有遇到最优的时候，一直相当于len(self.actions)分之一的epsilon累加
            #再看下我们对epsilon的设置和他本身表示的意义：我们一般把它设置的很小 然后表示我们很倾向于选择最优的那种情况
            #也就是说epsilon本身都很小了，你还len(self.actions)分之一 让他们 大于r的可能性有点小
            #所以直到遇到最优的a的时候才会大于，这时候的i也就是最优动作的i，这时候我们就跳出函数
            #得到的动作a也就是最优动作
            #如果不小心r变得过大，最后最后都没有大于r那就直接找最后那个动作当做最优动作
            
            if s >= r: return self.actions[i]
        
        return self.actions[len(self.actions)-1]
