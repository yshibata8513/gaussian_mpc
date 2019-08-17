import torch

ql=1.
qc=1.
cv = 0.5

def construct_loss_(path,refs,vs,variances):
    mean_error = error_deviation_parallel_(path,refs,ql,qc)
    var_error = ( variances*(path[:,:2]-refs[:,:2]).pow(2).sum(dim=1) ).sum()
    return mean_error + cv*var_error


def error_deviation_parallel_(states,refs,ql,qc):
    x,y = states[:,0],states[:,1]
    cx,cy,gx,gy = refs[:,0],refs[:,1],refs[:,2],refs[:,3]
    el = -gx*(x-cx) - gy*(y-cy)
    ec =  gy*(x-cx) - gx*(y-cy)
    err = ql*el.pow(2).sum()+qc*ec.pow(2).sum()
    return err


def search_(state,waypoints,_indx,L):
    x,y = state[0],state[1]
    cx,cy= waypoints[:,0],waypoints[:,1]
    dx = x-cx[_indx:_indx+L]
    dy = y-cy[_indx:_indx+L]
    d2 = dx.pow(2) + dy.pow(2)
    indx = torch.argmin(d2) + _indx
    return indx

