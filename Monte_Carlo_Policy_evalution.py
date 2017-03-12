# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#state_sample, action_sample, reward_sample 分别是状态、动作和奖励系列
def mc(gamma, state_sample, action_sample, reward_sample):   
    vfunc = dict();
    nfunc = dict();
#蒙特卡洛算法里面的S(s) 和 N（s）函数 用以建立字典
    for s in states:
        vfunc[s] = 0.0
        nfunc[s] = 0.0 
#每个S(s) 和 N（s）中元素个数都和状态数一样，初始化归零
    for iter1 in xrange(len(state_sample)):
     """
        这个iter代表的意思应该是以不同状态s起始的意思
        后面的step表示选定了一个起始状态，然后依次移动到一个新的状态
        所以确定初始状态的还是借助这个，毕竟只按照一个状态当初始没有说明性
        所以遍历一遍
        """
        #后面S(s) 和 N（s）都需要累加迭代
        #迭代的次数和那一个状态-动作-回报串中状态数目一样
        G = 0.0
        for step in xrange(len(state_sample[iter1])-1,-1,-1):
            #从状态数目减一 开始 到0为止（包括0）（总共还是状态数目那么多）
            #每次迭代都需要遍历一遍全部的例子
            
            G *= gamma;
            #G就是我们蒙特卡洛里面说的说的gs 
            #计算方法就是从开始那个状态s的回报
            #到s下一个状态s‘看它的回报乘上一个衰减因子gamma累加
            #到下下个状态s‘’的话，s‘’的回报就需要乘上gamma的平方了，以此类推
            G += reward_sample[iter1][step];
            #这也是这里先乘后加的原因，毕竟需要gamma的只是
            #之后的状态的，第一个状态不需要的 于是也就先加了
            #越靠前的需要的gamma次幂越少，于是就先加
            """
            gs代表的意思从某个状态开始 衰减奖励的累加，
            于是最终状态就只剩自己本身的奖励，而最开始的
            状态式子才是最长；
            不清楚的话可以列举下一个迭代9次 每次有9个状态的
            每次每个状态的gs应该怎么列式子
            我们是从最短的开始算，也就是从gs8的情况开始，这时候只有一个
            而g's7就有2个了以此类推
            """
        for step in xrange(len(state_sample[iter1])):
            s         = state_sample[iter1][step]
            vfunc[s] += G;
            nfunc[s] += 1.0;
        #计算S(s) 和 N（s），s不但包含了状态还包含了迭代次数
            G        -= reward_sample[iter1][step]
"""
            这里遍历每个状态开始的时候是从最开始那个状态开始的
            意思也就是描述那个状态回报包含的式子是最长的
            最长的那个比如说是gs0，减去最开始的那个r0
            然后再除上个gamma，就是gs1的样子，不清楚的再列一下式子
"""
            G        /= gamma;


    for s in states:
        if nfunc[s] > 0.000001:
            vfunc[s] /= nfunc[s]
#判定是否收敛
