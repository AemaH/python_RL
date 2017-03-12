# -*- coding: utf-8 -*-
def sarsa(num_iter1, alpha, epsilon):
    #同样和 MC Control 算法一样，
    # SARSA 的状态-动作价值也收敛到 ϵ− 贪婪策略的状态-动作价值上。
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0
#q和设置key全部的可能情况

    for iter1 in xrange(num_iter1):
        s = states[int(random.random() * len(states))]
        a = actions[int(random.random() * len(actions))]
        #初始化开始的状态和动作
        #同样的他们的回报r也还是依照前面预设的那些r
        t = False
        while False == t:
            key         = "%d_%s"%(s,a)
            t,s1,r      = grid.transform(s,a)
            a1          = epsilon_greedy(s1, epsilon)
            key1        = "%d_%s"%(s1,a1)
            qfunc[key]  = qfunc[key] + alpha * ( \
                          r + gamma * qfunc[key1] - qfunc[key])
            #这里和 MC Control的不同又体现出来了
            #MC Control求取q(s,a)的方式是累加比值
            #换句话说每次就按一个状态-动作对 都需要累加先前g
           #所以需要保留每一个 s r a，而SARSA不同，他只需要q就行
           #就可以满足接下来计算了，
           #所以啊~程序的设计还是先要看之前的式子里要怎么做
            s           = s1
            a           = a1