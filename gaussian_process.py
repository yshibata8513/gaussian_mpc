import torch

sigma = torch.Tensor([0.2])

noise = 0.1



def RBF_kernel(x,x_,dim):
    x = x.view(-1,1,dim)
    x_ = x_.view(1,-1,dim)
    dist = (x-x_).pow(2).sum(dim=-1)
    return torch.exp(-dist/2/sigma.pow(2))
    
    
class GaussianProcess:
    
    def __init__(self,dim_x,dim_y,var,kernel=None):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.var = var
        if kernel:
            self.kernel = kernel
        else:
            self.kernel = RBF_kernel
            
    def train(self,x,y):
            self.train_size = x.size()[0]
            self.train_x = x
            self.train_y = y
            self.Knn = self.kernel(x,x,self.dim_x)
            self.In = torch.eye(self.train_size)
            self.G = (self.Knn + self.var*self.In).inverse()
            
            
    def predict_(self,pseudo_x,pred_x):
            Kmm = self.kernel(pseudo_x,pseudo_x,self.dim_x)
            Kmm_inv = Kmm.inverse()
            Knm = self.kernel(self.train_x,pseudo_x,self.dim_x)
            Kmn = Knm.t()
            Kpm = self.kernel(pred_x,pseudo_x,self.dim_x)
            Kmp = Kpm.t()
            
            _L =  self.Knn - Knm@Kmm_inv@Kmn
            L = _L*I
            print(L)
            
            F = (L + self.var*self.In).inverse()
            
            Q = Kmm + Kmn@F@Knm
            Q_inv = Q.inverse()
            
            #print(Kpm.size(),Q_inv.size(),Kmn.size(),F.size(),self.train_y.size())
            
            myu_p = Kpm@Q_inv@Kmn@F@self.train_y
                    
            return myu_p

    
    def forward(self,pred_x,requires_sigma=False):
        Kpp = self.kernel(pred_x,pred_x,self.dim_x)
        Kpn = self.kernel(pred_x,self.train_x,self.dim_x)
        Knp = Kpn.t()
        
        myu_p = (Kpn@self.G@self.train_y).view(-1)
        sigma_p = None
        if requires_sigma:
            sigma_p = Kpp - Kpn@self.G@Knp
            sigma_p = torch.eye(self.dim_y)*sigma_p        
        return myu_p,sigma_p
        
    
    def distance_measure(self):
        Kxn = self.kernel(x,self.train_x)
        Knx = Kxn.t()
        Kxx = self.kernel(x,self.train_x)
        return Kxx - Kxn@self.Knn_inv@Knx