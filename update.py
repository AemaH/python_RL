# -*- coding: utf-8 -*-
#qfunc 是最优策略的 q 值
#alpha 是学习率
def update(policy, f, a, qfunc, alpha):
    pvalue        = policy.qfunc(f, a);
    error         = pvalue - tvalue; 
    fea           = policy.get_fea_vec(f, a);
    policy.theta -= alpha * error * fea; 
    
   """
   没写完的备注
   毕竟现在的时候没有状态s了
    因而我们现在使用的是f
    get_fea_vec()函数里面
        def get_fea_vec(self, fea, a):
        f = np.array([0.0 for i in xrange(len(self.theta))]);
            
        idx = 0
        for i in xrange(len(self.actions)):
            if a == self.actions[i]: idx = i;    
        for i in xrange(len(fea)):
            f[i + idx * len(fea)] = fea[i];
            
            参照可知，是用于建立f的函数
            f首先建立一行全为0的矩阵...
    """