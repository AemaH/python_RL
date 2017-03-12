# -*- coding: utf-8 -*-
"""
policy 待更新的策略
//f      状态特征
//a      动作
//qvalue q值
alpha  学习率
"""
def update_softmaxpolicy(policy, f, a, qvalue, alpha):

    fea  = policy.get_fea_vec(f,a);
    prob = policy.pi(f);
    
    delte_logJ = fea;
    for i in xrange(len(policy.actions)):
        a1          = policy.actions[i];
        fea1        = policy.get_fea_vec(f,a1);
        delta_logJ -= fea1 * prob[i];

    policy.theta -= alpha * delta_logJ * qvalue;  
        
  """
  这时候我们可以拿出先前的有关ε-greedy的策略更新update来对比着看
  分析每一部分：
      我们首先先看里面这个for循环，每个迭代对于某个动作在对应的位置放上
      某一状态s包含特征向量，当然这时候的f是函数开始就给定的，所以就别想了；
      然后得到delta_logJ，看名字也知道啊，就是那个softmax函数对数梯度的计算公式啊
      计算公式里需要对全部的动作a的某两个式子累计求和（不要在意之前的负号）
      看下policy_value.py可以知道pi()函数就是先前求的π（s，a）
          这个函数只需要丢入f，也就是对于的状态的特征；然后就看他对全部的a来求对每个a
          的概率；
      
"""      
        
        
    
        
        