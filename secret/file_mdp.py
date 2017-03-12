# -*- coding: utf-8 -*-
#/bin/python
import numpy;
import random;

class Grid_Mdp:

    def __init__(self):

        self.states            = [1,2,3,4,5,6,7,8] # 0 indicates end
        self.terminal_states      = dict()
        self.terminal_states[6]   = 1
        self.terminal_states[7]   = 1
        self.terminal_states[8]   = 1

        self.actions        = ['n','e','s','w']

        self.rewards        = dict();
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        self.t              = dict();
        self.t['1_s']       = 6
        self.t['1_e']       = 2
        self.t['2_w']       = 1
        self.t['2_e']       = 3
        self.t['3_s']       = 7
        self.t['3_w']       = 2
        self.t['3_e']       = 4
        self.t['4_w']       = 3
        self.t['4_e']       = 5
        self.t['5_s']       = 8 
        self.t['5_w']       = 4

        self.gamma          = 0.8

    def getTerminal(self):
        return self.terminal_states;

    def getGamma(self):
        return self.gamma;    

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def transform(self, state, action): ##return is_terminal,state, reward
        if state in self.terminal_states:
            return True, state, 0

        key = '%d_%s'%(state, action);
        if key in self.t: 
            next_state = self.t[key]; 
        else:
            next_state = state       
 
        is_terminal = False
        if next_state in self.terminal_states:
            is_terminal = True
      
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key];
           
        return is_terminal, next_state, r;




    def gen_randompi_sample(self, num):
    #设置迭代次数num
        state_sample  = [];
        action_sample = [];
        reward_sample = [];
        #设置成list列表数据类型
        for i in xrange(num):
                s_tmp = []
                a_tmp = []
                r_tmp = []
                #设置成list列表数据类型

                s = self.states[int(random.random() * len(self.states))]
                #随机设置初始状态 毕竟states为8乘上个从0到1的随机数，然后取整确定一个初始状态
                t = False
                while False == t:
                    a = self.actions[int(random.random() * len(self.actions))]
                    #动作a和状态s一样，随便的取一个动作；
                    #这里选取a的策略就是随机选取，因而可以这样写
                    #一般情况下还有动作a的选取要求条件，依照那个条件设定来选取
                    t, s1, r  = self.transform(s,a)
                    """将上面随便设置的初始状态和动作输入进去后
                    用于判断状态然后判定动作 接着确定接下来的状态；
                    接下来这个循环下面的语句，然后回到while这里，
                    再随机选取一个动作
                    如果是最终状态，transform函数会对应t那个变量输出True
                    也就跳出while这个循环
                    """
                    s_tmp.append(s)
                    r_tmp.append(r)
                    a_tmp.append(a)
                    #利用append()函数依次在s列表 r列表 a列表 输入每次得到的状态s 回馈r 动作a
                    s = s1  
                    #将s替换成上面选取的动作a对应的状态          
                state_sample.append(s_tmp)
                reward_sample.append(r_tmp)
                action_sample.append(a_tmp)
                #完成一个循环后，到达最终状态后，将这一次循环的情况输入sample表示一次采样

        return state_sample, action_sample, reward_sample

