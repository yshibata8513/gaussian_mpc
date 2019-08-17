import torch
import torch.nn as nn
from evals import *
from optimization import *
from gauss_update import *
from kinetic_model import *
import cubic_spline_planner


import matplotlib.pyplot as plt
import numpy as np

ax = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]
ay = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]
goal = [ax[-1], ay[-1]]

cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.2)

cx = torch.Tensor(cx).view(-1,1)
cy = torch.Tensor(cy).view(-1,1)

gx = cx[1:]-cx[:-1]
gy = cy[1:]-cy[:-1]


waypoints = torch.cat([cx[:-1],cy[:-1],gx,gy],dim=1)

start=150

x0,y0,gx0,gy0 = waypoints[start]
v0 = torch.sqrt(gx0.pow(2)+gy0.pow(2))*3
yaw0 = torch.atan2(gy0,gx0)
delta0=torch.zeros(1)
a0=torch.zeros(1)
state0 = torch.cat(  [x0.view(1),y0.view(1),yaw0.view(1),v0.view(1),delta0,a0])
control0 = torch.Tensor([0.0001,0.0001,0.001])



len_horizon = 10
EMAX=2000
kappa=0.01
TMAX = 1

step = 0
_controls = torch.nn.Parameter(torch.zeros(len_horizon,2))

dt = torch.ones(len_horizon,1)
vs = torch.nn.Parameter(torch.ones(len_horizon)).data.clone()*3.0001


print(vs)
opt = torch.optim.Adam([vs]+[_controls],lr=0.01)
#print(controls.size())
for T in range(TMAX):
    if T>0:
        state0 = s2_[0]
        _controls = torch.nn.Parameter(controls2_.data.clone())
        start = start2_
    for epoch in range(EMAX):
        controls = torch.cat([_controls,dt],dim=1)
        s=state0
        s_=[]
        for t in range(len_horizon):
            s = model(s,controls[t])
            s_.append(s.view(1,-1))
        s_ = torch.cat(s_,dim=0)
        if epoch==0:
            s0_ = s_.data.clone()
        dev = deviation_error_.apply
        dev_loss = dev(s_,vs,waypoints,start)
        v_loss = -kappa*vs.sum()
        loss = dev_loss + v_loss
        if epoch % 100 ==0:
            print(dev_loss.data.numpy(),v_loss.data.numpy())
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
    
    
    s2_ = s_[step:]
    x_ = s2_[:,0].data.numpy()
    y_ = s2_[:,1].data.numpy()

    controls2_ = _controls.data.clone()
    controls2_[:-(step+1)] = _controls[step+1:]
    controls2_[-(step+1):]=0
    start2_ = start + (step +1)*3