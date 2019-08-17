import torch

learning_rate=0.01
dim_s=2
EMAX=1000

class GP_MPC:
    def __init__(self,gp_propagator,evaluate,len_horizon,waypoints):
        self.waypoints = waypoints
        self.len_horizon = len_horizon
        self.gp_propagator = gp_propagator
        self.evaluate = evaluate
        
    def run(self,state_init,controls_init,vs,dt,start,_vars):
        start = torch.LongTensor([start])
        state_init = state_init.data.clone()
        controls = torch.nn.Parameter(controls_init.data.clone())
        vs = torch.LongTensor(vs.data.clone())
        if len(dt)==1:
            dt = torch.ones(self.len_horizon).view(-1,1)*dt
        opt =  torch.optim.Adam([controls],lr=learning_rate)
        refs = self.waypoints[vs[1:]+start]
        _,sigma_init = self.gp_propagator.initialize(state_init,controls[0])

        for epoch in range(EMAX):
            controls_dt = torch.cat([controls,dt],dim=1)
            state  = state_init
            
            path = []
            if epoch == EMAX-1:
                vars_ = []
                sigma = sigma_init
                for t in range(self.len_horizon):
                    state,sigma = self.gp_propagator.forward(state,controls_dt[t],sigma,True)
                    sigma_xy = sigma[:dim_s,:dim_s]
                    _,eigs,_ = sigma.svd()
                    var = eigs[0]                   # the largest eigenvalue of sigma_xy 
                    path.append(state.view(1,-1))
                    vars_.append(var.view(1))
                path = torch.cat(path,dim=0)
                vars_ = torch.cat(vars_,dim=0)
            else:
                for t in range(self.len_horizon):
                    state,_ = self.gp_propagator.forward(state,controls_dt[t],None,False)
                    path.append(state.view(1,-1))
                path = torch.cat(path,dim=0)
            opt.zero_grad()
            loss = self.evaluate(path,refs,vs,_vars) 
            loss.backward(retain_graph=True)
            opt.step()
            opt.zero_grad()
            '''
            if epoch % 500 ==0:
                print(loss.data.numpy())
            '''
        controls_dt = torch.cat([controls,dt],dim=1)
        return controls_dt,path,vars_.data.clone()
    
    
    
    
class MPC:
    def __init__(self,model,evaluate,len_horizon,waypoints):

        self.waypoints = waypoints
        self.len_horizon = len_horizon
        self.model = model
        self.evaluate = evaluate

        
    def run(self,state_init,controls_init,vs,dt,start):
        start = torch.LongTensor([start])
        state_init = state_init.data.clone()
        controls = torch.nn.Parameter(controls_init.data.clone())
        vs = torch.LongTensor(vs.data.clone())
        if len(dt)==1:
            dt = torch.ones(self.len_horizon).view(-1,1)*dt
        opt =  torch.optim.Adam([controls],lr=learning_rate)
        refs = self.waypoints[vs[1:]+start]

        for epoch in range(EMAX):
            controls_dt = torch.cat([controls,dt],dim=1)
            state  = state_init
            path = []
            for t in range(self.len_horizon):
                state = self.model(state,controls_dt[t])
                path.append(state.view(1,-1))
            path = torch.cat(path,dim=0)
     
            opt.zero_grad()
            
            loss = self.evaluate(path,refs,vs) 
            loss.backward()
            opt.step()
            if epoch % 500 ==0:
                print(loss.data.numpy())
        controls_dt = torch.cat([controls,dt],dim=1)
        return controls_dt,path    