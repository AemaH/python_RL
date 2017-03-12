# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#td 算法也可以输入状态-动作-奖励序列。
def td(alpha, gamma, state_sample, action_sample, reward_sample):
    #TD算法包含的参数，参照那个式子 包含alpha gamma 状态 动作 回馈
    vfunc = dict()
    #对回报函数建立字典
    for s in states:
        vfunc[s] = 0.0 
#对每个状态遍历 对应参数 初始化归零          
 
    for iter1 in xrange(len(state_sample)):
     """
        这个iter代表的意思应该是以不同状态s起始的意思
        后面的step表示选定了一个起始状态，然后依次移动到一个新的状态
        所以确定初始状态的还是借助这个，毕竟只按照一个状态当初始没有说明性
        所以遍历一遍
        """
        #毕竟不可能只选一个状态按照这个策略
        #参照蒙特卡洛算法和策略迭代，不可能只算一次序列（一系列状态动作回报序列）
        #就完事，多来几次 求平均值才能有所保证
        
        for step in xrange(len(state_sample[iter1])):
            #对一次迭代 需要对全部的状态都更新
            s = state_sample[iter1][step]
            r = reward_sample[iter1][step]
            #于是就像一个二维的坐标一样 分别填入相关的s和对应的奖励r
            if len(state_sample[iter1]) - 1 > step:
                #step都是来自于xrange(len(state_sample[iter1]这个生成器的
                #参照下面的解释
                s1 = state_sample[iter1][step+1]
"""
毕竟s1代表的是next_s 也是这一个系列的状态中，所以step最多只能到最后面减1的位置
不然出现个状态在这一系列状态之外也就很尴尬了
所以设置个  if len(state_sample[iter1]) - 1 > step:
    当len(state_sample[iter1]) - 1 >= step 当然也不会存在
    他们只会被设置为0
    最后那个就在len(state_sample[iter1]) - 2 = step
    的时候被设置了，例如len(state_sample[iter1])=9
    那么state_sample[iter1][8] 就在state_sample[iter1][7]的时候
    利用 s1 = state_sample[iter1][step+1]设置好了
    同样的这时候的v(s) 也被设置

"""
                next_v = vfunc[s1]
            else:
                next_v = 0.0;

            vfunc[s] += alpha * (r + gamma * next_v - vfunc[s]); 
#TD算法的式子；初始状态的回报就是r回馈奖励 下一个状态才会被利用他
#毕竟最开始使用这个式子的时候，也是对应最开始状态s的next_s了
