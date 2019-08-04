import torch

def error_deviation_parallel(states,waypoints,indices,ql,qc):
    x,y = states[:,0],states[:,1]
    cx,cy,gx,gy = waypoints[indices,0],waypoints[indices,1],waypoints[indices,2],waypoints[indices,3]
    el = -gx*(x-cx) - gy*(y-cy)
    ec =  gy*(x-cx) - gx*(y-cy)
    err = ql*el.pow(2).sum()+qc*ec.pow(2).sum()
    return err


def error_deviation(state,waypoints,indx):
    x,y,phi,vx,vy,r,delta,T = state
    cx,cy,gx,gy = waypoints[indx]   
    el = -gx*(x-cx) - gy*(y-cy)
    ec =  gy*(x-cx) - gx*(y-cy)
    return el,ec


kappa = 0.1

def error_velocity(inds,ds,horizon_length):
    
    return -kappa*ds*horizon_length*(inds[-1]-inds[0])

L = 100

def search_(state,waypoints,_indx):
    x,y,_,_,_,_,_,_ = state
    cx,cy,_,_ = waypoints
    dx = x-cx[_indx:_indx+L]
    dy = y-cy[_indx:_indx+L]
    d2 = dx.pow(2) + dy.pow(2)
    indx = torch.argmin(d2) + _indx

    return indx_




def search_parallel(states,waypoints,_indx):
    xs,ys = states[:,0],states[:,1]
    cx,cy = waypoints[:,0],waypoints[:,1]


def error_deviation_parallel_(states,refs,indices,ql,qc):
    x,y = states[:,0],states[:,1]
    cx,cy,gx,gy = refs[:,0],refs[:,1],refs[:,2],refs[:,3]
    el = -gx*(x-cx) - gy*(y-cy)
    ec =  gy*(x-cx) - gx*(y-cy)
    err = ql*el.pow(2).sum()+qc*ec.pow(2).sum()
    return err
    
    

ql=1
qc=1

class deviation_error(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,states,vs,waypoints):
        states = states.data.clone()
        states.requires_grad = True
        
        inds = vs.round().long()
        
        dw = waypoints[inds+1]-waypoints[inds]
        
        refs = waypoints[inds].data.clone().requires_grad_()
        with torch.enable_grad():
            error = error_deviation_parallel_(states,refs,inds,ql,qc)
            de_s,de_w = torch.autograd.grad(error,[states]+[refs],grad_outputs=None,retain_graph=False,create_graph=False)
 
        ctx.save_for_backward(states.data.clone(),vs.data.clone(),waypoints,de_s.data.clone(),de_w.data.clone(),dw.data.clone())
        
        return error
    
    @staticmethod
    def backward(ctx,de):
        states,vs,waypoints,de_s,de_w,dw = ctx.saved_tensors
         
        de_theta = (de_w@dw.t()).diag().view(-1,1)
        
        _range = torch.range(0,len(states)-1).view(-1,1)
        range_ = _range.t()
        mask = (_range<=range_).float()
        de_v = (mask@de_theta).view(-1)
   
        return de_s,de_v,None