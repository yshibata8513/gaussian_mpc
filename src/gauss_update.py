import torch

noise = 0.1
sigma_w = 0.01

dim_x = 2
dim_v = 2
dim_a = 2 
dim_z = dim_v + dim_a  #dimension of GP input 
dim_s = dim_x + dim_v + dim_a

dim_c = 2

v2s = torch.cat([torch.zeros(dim_x,dim_v),torch.eye(dim_v),torch.zeros(dim_a,dim_v)],dim=0)
s2v = v2s.t()
z2s = torch.cat([torch.zeros(dim_x,dim_z),torch.eye(dim_z)],dim=0)
s2z = z2s.t()


class GP_test:
    def __init__(self,dim):
        self.dim = dim
    
    def forward(self,state,control,requires_sigma=False):
        myu = torch.ones(self.dim)*state[0]*0.0001
        sigma = None
        if requires_sigma:
            sigma = torch.eye(self.dim)*noise
        return myu,sigma

class gaussian_propagator:
    def __init__(self,p_model,gp_model,sigma_w):
        self.p_model = p_model
        self.gp_model = gp_model
        self.sigma_w = sigma_w 

    def initialize(self,state,control):
        z = (s2z@state)
        #state = torch.nn.Parameter( state ).view(-1)
        control = control.data.clone().view(-1)
        gp_indata = torch.cat([z,control],dim=0)
     
        myu,sigma = self.gp_model(gp_indata,requires_sigma=True)
     
        return v2s@myu,v2s@sigma@s2v

    def forward(self,state,control_dt,_sigma,requires_sigma=False):
        control = control_dt[:dim_c]
        z = (s2z@state)
        #state = torch.nn.Parameter( state ).view(-1)
        #control = control.data.clone().view(-1)
        gp_indata = torch.cat([z,control],dim=0)
        f = self.p_model(state,control_dt)
        g,_ = self.gp_model(gp_indata,requires_sigma=False)
        myu = f + v2s@g
        sigma = None 
        if requires_sigma:
            df = self.differentiate(state,control_dt,type='p')
            dg = self.differentiate(state,control,type='gp')
            dg = dg@s2z
            sigma = self.construct_sigma(df,dg,_sigma,sigma_w)
        return myu,sigma
        
    def construct_sigma(self,df,dg,sigma_x,sigma_w):
        sigma_00 = sigma_x
        sigma_10 = dg@sigma_x
        sigma_01 = sigma_10.t()
        sigma_11 = sigma_10@dg.t() + sigma_w
        
        sigma_0 = torch.cat([sigma_00,sigma_01],dim=1)
        sigma_1 = torch.cat([sigma_10,sigma_11],dim=1)
        
        sigma_ = torch.cat([sigma_0,sigma_1],dim=0)
        vec = torch.cat([df,v2s],dim=1)
        sigma = vec@sigma_@vec.t()
        
        return sigma
        
    def differentiate(self,state,control,type):
        if type == 'gp':
            state = (s2z@state.data.clone())
            state = torch.nn.Parameter( state ).view(-1)
            control = control.data.clone().view(-1)
            gp_indata = torch.cat([state,control],dim=0)
            F,_ = self.gp_model(gp_indata,requires_sigma=False)

        elif type=='p':
            state = torch.nn.Parameter( state.data.clone() )
            control = control.data.clone()
            F = self.p_model(state,control)

        with torch.enable_grad():
                dF_ = [torch.autograd.grad(f,[state],grad_outputs=None,retain_graph=True,create_graph=False,only_inputs=True,allow_unused=True)[0].view(1,-1) for f in F]
                dF  = torch.cat(dF_,dim=0)

        return dF.data.clone()

