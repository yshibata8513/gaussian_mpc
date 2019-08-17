import torch


THRESHOLD = -0.01
learning_rate = 0.001
EMAX=100000

class constrained_optimization:
    def __init__(self,obj,cons,myu):
        self.obj = obj
        self.cons = cons
        self.myu = myu
    
    def compute_penalty(self,x):
        penalty = 0
        for con in self.cons:
            c = torch.max(con(x),THRESHOLD*torch.ones_like(con(x)))
            #if c>THRESHOLD:
            penalty += torch.log(-c).sum()
        return penalty
    
    def compute_cost(self,x):
        cost = self.obj(x) - self.myu*self.compute_penalty(x)
        return cost
    
    def judge_bleaching(self,x):
        judge = torch.prod(torch.cat([torch.prod(c(x)<=0).view(1) for c in self.cons]) )
        return judge

    def run(self,x0):
        c=0
        x = torch.nn.Parameter(x0)
        opt=torch.optim.Adam([x],lr=learning_rate)
        _x = x.data.clone()
        for epoch in range(EMAX):
       
            cost = self.compute_cost(x)
            opt.zero_grad()
            cost.backward()
            opt.step()
            #print(x)
            #print(self.judge_bleaching(x))
            if self.judge_bleaching(x):
                _x = x.data.clone()
            else:
                return _x
        return x