# -*- coding: utf-8 -*-
def qlearning(num_iter1, alpha, epsilon):
   #首先我们来确定里面的式子
   #不可能只选一个起始状态，所以我们要迭代
   #有alpha 有epsilon 所以都要设置，函数都需要被给予
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0
        #初始归零
    for iter1 in xrange(num_iter1):

        s = states[int(random.random() * len(states))]
        a = actions[int(random.random() * len(actions))]
        #参照SARSA里面的初始状态和动作都需要初始化
        t = False
        while False == t:
            key         = "%d_%s"%(s,a)
            t,s1,r      = grid.transform(s,a)
            #根据上面得到的key，转化为下一个状态
            key1 = ""
            qmax = -1.0
            #最开始q至少为0，所以肯定比这大，
            #所以可以让接下来的继续下去，其实你只要设置个负数都行
            for a1 in actions:
                #从action集合里面找下一动作a'
                #找到那个让q最大动作a，然后更新当做当前的key1
                if qmax < qfunc["%d_%s"%(s1,a1)]:
                    qmax = qfunc["%d_%s"%(s1,a1)]
                    key1 = "%d_%s"%(s1,a1)
            qfunc[key]  = qfunc[key] + alpha * ( \
                          r + gamma * qfunc[key1] - qfunc[key])
            #上面那个式子就是qlearning的定义式
            s           = s1
            a           = epsilon_greedy(s1, epsilon)
            #到达了新的状态s1了，设置成下一个要使用的s
            #注意！！！！
                #这里的a选取的不同也是Qlearning和SARSA最大的不同
                #SARSA里面直接就是用上面式子的a1（当然式子也不一样）了
                #而Qlearning里面选取动作的策略和值函数更新的策略不同
                # 称作离策略（off-policy）
   