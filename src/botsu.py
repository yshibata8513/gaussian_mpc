
'''


def construct_loss(path,refs,vs,variances):
    mean_error = error_deviation_parallel_(path,refs,ql,qc)
    var_error = ( variances*(path[:,:2]-refs[:,:2]).pow(2).sum(dim=1) ).sum()
    return mean_error + cv*var_error


def error_deviation_parallel(states,waypoints,indices,ql=1.,qc=1.):
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




def search_parallel(states,waypoints,_indx):
    xs,ys = states[:,0],states[:,1]
    cx,cy = waypoints[:,0],waypoints[:,1]
    return 0

ql=1.
qc=1.

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


class deviation_error_(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,states,vs,waypoints,start):
        states = states.data.clone()
        states.requires_grad = True
        
        _range = torch.arange(len(states)).view(-1,1)
        range_ = _range.t()
        mask = (_range<=range_).float()
        
        inds = (mask.t()@vs.view(-1,1)).view(-1).round().long()+start
        #print(inds)
        #print([vs[:i+1].sum().data+start for i in range(len(vs))])
        dw = waypoints[inds+1]-waypoints[inds]
        
        refs = waypoints[inds].data.clone().requires_grad_()
        with torch.enable_grad():
            error = error_deviation_parallel_(states,refs,inds,ql,qc)
            de_s,de_w = torch.autograd.grad(error,[states]+[refs],grad_outputs=None,retain_graph=False,create_graph=False)
 
        ctx.save_for_backward(states.data.clone(),vs.data.clone(),waypoints,de_s.data.clone(),de_w.data.clone(),dw.data.clone())
        #print(states.size(),vs.size(),waypoints.size(),de_s.size(),de_w.size(),dw.data.size())
        
        return error.data.clone()
    
    @staticmethod
    def backward(ctx,de):
        states,vs,waypoints,de_s,de_w,dw = ctx.saved_tensors
         
        de_theta = (de_w@dw.t()).diag().view(-1,1)
        _range = torch.arange(len(states)).view(-1,1)
        range_ = _range.t()
        mask = (_range<=range_).float()
        
        de_v = (mask@de_theta).view(-1)
        #print(de_v)
        return de_s.data.clone(),de_v.data.clone(),None,None
        
        
        
        
'''