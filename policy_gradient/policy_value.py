# -*- coding: utf-8 -*-
#!/bin/python
import sys
sys.path.append("./secret")

import grid_mdp
import random
random.seed(0)
import numpy as np


class SoftmaxPolicy:
    def __init__(self, grid, epsilon):
        self.actions = grid.actions

        grid.start();
        t,hats,r = grid.receive(self.actions[0]);
        self.theta = [ 0.0  for i in xrange(len(hats)*len(self.actions)) ]
        self.theta = np.array(self.theta);
        self.theta = np.transpose(self.theta);
        
        self.epsilon = epsilon

    def get_fea_vec(self, fea, a):
        f = np.array([0.0 for i in xrange(len(self.theta))]);
            
        idx = 0
        for i in xrange(len(self.actions)):
            if a == self.actions[i]: idx = i;    
        for i in xrange(len(fea)):
            f[i + idx * len(fea)] = fea[i];
        
        return f

    def pi(self, fea):
        prob = [ 0.0 for i in xrange(len(self.actions))];
        sum1 = 0.0;
        for i in xrange(len(self.actions)):
            f = self.get_fea_vec(fea,self.actions[i])
            prob[i] = np.exp(np.dot(f, self.theta));
            sum1 += prob[i];
            #这个累加是为了softmax策略里面的分母部分需要对全部动作求和得到一个数
            #然后下面的对每个数除以这个和，也就是softmax里面每个动作的概率
        for i in xrange(len(self.actions)):
            prob[i] /= sum1;

        return prob;

    def take_action(self, fea):
        prob = self.pi(fea);

        ##choose
        r = random.random()
        s = 0.0
        for i in xrange(len(self.actions)):
            s += prob[i]
            if s >= r: return self.actions[i];
        
        return self.actions[len(self.actions)-1];
        #这里的两个return的原因分别是：首先借助pi(fea)我们知道了
        #fea这个代表的状态对应全部动作每个被选取的可能性，然后随机设置非零数r
        #然后对概率累加；
        #这时候我们要注意一点就是上面概率的计算方法，采用的softmax函数的原因：
            #softmax特点就是如果某一个zj大过其他z,那这个映射的分量就逼近于1,其他就逼近于0
        #当我们到最优动作的时候，这时候因为softmax的原因会让这个概率足够大，所以也就会出现其他动作的概率都很小
        #累加半天都没有随机设置的随机数r大，然后出现最优的时候，一下子就超过了r 然后我们就知道 哦 这个是最优动作
        #其实想法和ε-greedy设置最优动作对应概率最大的想法一样；
        #然后同样的，如果不小心随机的r很大，直到最后都没出现超过它的，那只能把最后一个当做最大了（摊手）


class ValuePolicy:
    def __init__(self, grid, epsilon):
        self.actions = grid.actions

        grid.start();
        t,hats,r = grid.receive(self.actions[0]);
        self.theta = [ 0.0  for i in xrange(len(hats)*len(self.actions)) ]
        self.theta = np.array(self.theta);
        self.theta = np.transpose(self.theta);
        
        self.epsilon = epsilon

    def get_fea_vec(self, fea, a):
        f = np.array([0.0 for i in xrange(len(self.theta))]);
            
        idx = 0
        for i in xrange(len(self.actions)):
            if a == self.actions[i]: idx = i;    
        for i in xrange(len(fea)):
            f[i + idx * len(fea)] = fea[i];
        
        return f

    def qfunc(self, fea, a):
        f = self.get_fea_vec(fea, a);
        return np.dot(f, self.theta);


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

        ##choose
        r = random.random()
        s = 0.0
        for i in xrange(len(self.actions)):
            s += pro[i]
            if s >= r: return self.actions[i]
        
        return self.actions[len(self.actions)-1]
