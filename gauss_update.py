import torch


def construct_sigma(df,dg,sigma_x,sigma_w,B):
    
    sigma_00 = sigma_x
    sigma_10 = dg@sigma_x
    sigma_01 = sigma_10.t()
    sigma_11 = sigma_10@dg.t() + sigma_w
    
    sigma_0 = torch.cat([sigma_00,sigma_01],dim=1)
    sigma_1 = torch.cat([sigma_10,sigma_11],dim=1)
    
    sigma_ = torch.cat([sigma_0,sigma_1],dim=0)
    
    vec = torch.cat([df,B],dim=1)
    
    sigma = vec@sigma_@vec.t()
    
    return sigma
    
    
def evaluate(func,state,control,grad=True):
    F = func(state,control)
    dF = None
    if grad:
        dF_ = [torch.autograd.grad(f,[state],grad_outputs=None,retain_graph=True,create_graph=False,only_inputs=True,allow_unused=True)[0].view(1,-1) for f in F]
        dF  = torch.cat(dF_,dim=0)
    return F,dF