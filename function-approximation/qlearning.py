# -*- coding: utf-8 -*-

#从一开始我们就可以看出来这里面不再包含epsilon 在后面是直接调用函数完成的
#同时也要注意这时候的状态也不一样了
#状态s不再是那一系列的 状态1 状态2等等
#而是一个个向量，状态1就是第一个数为1其余为0，以此类推

def qlearning(grid, policy, num_iter1, alpha):
    actions = grid.actions;
    gamma   = grid.gamma;
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1
#初始化全部的theta

    for iter1 in xrange(num_iter1):
        f = grid.start();   
        #从一个随机非终止状态开始， f 是该状态的特征
        #start()函数在grid.py里面用于得到初始化状态
        a = actions[int(random.random() * len(actions))]
        #初始化一个动作
        t = False
        count = 0
       #毕竟计算的时候需要上下行每一行的参数个数一致 
        while False == t and count < 100:
            t,f1,r      = grid.receive(a)
            #t  表示是否进入终止状态
            #f1 是环境接受到动作 a 之后转移到的状态的特征。
            #r  表示奖励

            qmax = -1.0
            for a1 in actions:
                pvalue = policy.qfunc(f1, a1);
#参照qlearning里面的吧f1看成s，然后这部分就是遍历所有动作
#然后找到让这个f1最大的那个a1，
#这里用qfunc只是初始化数值，然后后面才是依次更新
#联系后面的update用以更新theta；
#所以总的就是：
#遍历所有动作得到的代表状态-动作的矩阵f和参数theta找到最大的那一个
#然后代表此时的qmax 然后借助update()更新theta的数值；
#update()的目标也就是优化和最优策略的误差得到参数theta
#这时候的参数然后乘上让乘积最大的动作，就代表max q
                if qmax < pvalue:  qmax = pvalue;
            update(policy, f, a, r + gamma * qmax, alpha);

            f           = f1
            a           = policy.epsilon_greedy(f)
            # 下一个状态的a的选取依旧依照ε-greedy来得到
            count      += 1   
    return policy;