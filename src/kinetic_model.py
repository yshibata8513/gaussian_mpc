import torch
import torch.nn as nn


WB = 2.5  # [m]
NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]

class vehicle_model:
    def __init__(self,WB,noise=False):
        super(vehicle_model,self).__init__()
        self.WB = WB
        self.noise=noise
    def forward(self,state,control):
        x,y,yaw,v,delta,a = state
        da,d_delta,dt = control
        x_ = x + v*torch.cos(yaw)*dt
        y_ = y + v*torch.sin(yaw)*dt
        yaw_ = yaw + v/self.WB*torch.tan(delta)*dt
        v_ = v + a*dt
        delta_ = delta + d_delta*dt
        a_ = a + da*dt
    
        state_ = torch.cat( [x_.view(1),y_.view(1),yaw_.view(1),v_.view(1),delta_.view(1),a_.view(1)] ,dim=0)
    
        return state_

    


def model(state,control):
    x,y,yaw,v,delta,a = state
    da,d_delta,dt = control
    x_ = x + v*torch.cos(yaw)*dt
    y_ = y + v*torch.sin(yaw)*dt
    yaw_ = yaw + v/WB*torch.tan(delta)*dt
    v_ = v + a*dt
    delta_ = delta + d_delta*dt
    a_ = a + da*dt
    
    state_ = torch.cat( [x_.view(1),y_.view(1),yaw_.view(1),v_.view(1),delta_.view(1),a_.view(1)] ,dim=0)
    
    return state_


def model_(state,control):
    x,y,gx,gy,delta,a = state
    v=torch.sqrt(gx.pow(2)+gy.pow(2))
    yaw = torch.tan(gy/gx)
    da,d_delta,dt = control
    x_ = x + v*torch.cos(yaw)*dt
    y_ = y + v*torch.sin(yaw)*dt
    yaw_ = yaw + v/WB*torch.tan(delta)*dt
    v_ = v + a*dt
    delta_ = delta + d_delta*dt
    a_ = a + da*dt
    
    state_ = torch.cat( [x_.view(1),y_.view(1),yaw_.view(1),v_.view(1),delta_.view(1),a_.view(1)] ,dim=0)
    
    return state_





'''

Df=
Dr=
Cf=
Cr=
Bf=
Br=
Cm1=
Cr0=
Cr2=
Iz=
Ptv= 


def forward(state,control):
    x,y,phi,vx,vy,r,delta,T = state
    d_delta,dt = control
    
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    cos_d = torch.cos(delta)
    sin_d = torch.sin(delta)
    
    alpha_r = torch.atan((vy-lr*r)/vx )
    alpha_f = torch.atan((vy-lf*r)/vx - delta )
    
    Fry = Dr*torch.sin( Cr*torch.atan(B*alpha_r) )
    Fry = Df*torch.sin( Cf*torch.atan(B*alpha_f) )
    Fx = Cm2*T - Cr0 - Cr2*vx*vx
    r_target = delta*vx/(lf+lr)
    tau = (r_target-r)*Ptv
    
    dx = vx*cos_p - vy*sin_p
    dy = vx*sin_p + vy*cos_p
    d_phi = r
    dvx = (Fx - Ffy*sin_d + m*vy*r)/m
    dvy = (Fry - Ffy*cos_d - m*vx*r)/m
    dr = (Ffy*lf*cos_d - Fry*lr + tau)/lz
    
    x_     = x + dx*dt
    y_     = y + dy*dt
    phi_   = phi + d_phi*dt
    vx_    = vx + dvx*dt
    vy_    = vy + dvy*dt
    r_     = r + dr*dt
    delta_ = delta + d_delta
    T_     = T + dt
    
    state_ = [x_,y_,phi_,vx_,vy_,r_,delta_,T_]
    
    state_ = torch.cat( [s.view(1) for s in state_] , dim=0 )
    
    return state_
    
'''