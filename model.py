import numpy as np
import math
from scipy.integrate import odeint

class Model():
    def __init__(self, rtol=1e-12, atol=1e-12, n=100, x=0.0, dx=0.0, p=0.0, dp=0.0, p_lim=20.0, x_lim=0.5, Ts=0.01, lambda_x=1.0, lambda_p=1.0, 
                 I=1.67e-3, l=0.1, r=0.02, m=0.5, J=0.02e-3, b=272e-6, Ke=0.686, Kt=0.082, R=7.4, L=7.45e-3):
        
        self.I = I # Body moment of inertia (kgm²)
        self.l = l # Pole lenght (m)
        self.r = r # Whell radius (m)
        self.m = m # Pole mass (kg)
        self.J = J # Wheel moment of inertia (kgm²)
        self.b = b # Motor friction coefficient (Nms)
        self.Kt = Kt # Motor torque constant (Nm/A)
        self.Ke = Ke # Motor speed constant (Vs)
        self.R = R # Motor resistance (R)
        self.L = L # Motor inductance (H)
        self.g = 9.80665 # Gravity (m/s²)
        
        # Differential equations parameters
        
        self.Ax = ((self.r * self.m * self.l) ** 2) * self.g
        self.Bx = (self.I + self.m * (self.l ** 2)) * self.r * self.Kt
        self.Cx = (self.I + self.m * (self.l ** 2)) * self.b
        self.Dx = (self.I + self.m * (self.l ** 2)) * (self.r ** 2) * self.m * self.l
        
        self.Ap = (self.J + (self.r ** 2) * self.m) * self.m * self.g * self.l
        self.Bp = self.m * self.l * self.r * self.Kt
        self.Cp = self.m * self.l * self.b
        self.Dp = (self.m * self.l * self.r) ** 2
        
        self.E = (self.J + (self.r ** 2) * self.m) * (self.m * (self.l ** 2) + self.I)
        self.F = (self.r * self.m * self.l) ** 2
        
        if L == 0:
            L=1e-12
            
        self.Ai = 1/L
        self.Bi = self.R/L
        self.Ci = self.Ke/(L*self.r)
        
        # Simulation parameters
        
        self.Ts = Ts # Iteration Period (s)
        self.n = n # Steps per Period
        self.T = self.Ts / self.n # Simulation Period (s)
        self.rtol = rtol # Relative tolerance
        self.atol = atol # Absolute tolerance
        self.p_lim = p_lim * math.pi / 180.0 # Inclination limit to end simulation
        self.x_lim = x_lim # Position limit to end simulation
        self.lambda_x = lambda_x # Reward function parameter
        self.lambda_p = lambda_p # Reward function parameter
        
        # States

        self.v = 0.0 # (V)
        self.x = x # (m)
        self.dx = dx # dx/dt (m/s)
        self.p = p # theta (rad)
        self.dp = dp # dp/dt (rad/s)
        self.i = 0.0 # (A)
        self.t = 0.0 # (s)
        
        self.x_list = [self.x]
        self.dx_list = [self.dx]
        self.p_list = [self.p]
        self.dp_list = [self.dp]
        self.i_list = [self.i]
        self.t_list = [self.t]
        self.v_list = [self.v]
    
    def reset_var(self, x=0.0, dx=0.0, p=0.0, dp=0.0):
        self.v = 0.0
        self.x = x
        self.dx = dx
        self.p = p
        self.dp = dp
        self.i = 0.0
        self.t = 0.0
        
        self.x_list = [self.x]
        self.dx_list = [self.dx]
        self.p_list = [self.p]
        self.dp_list = [self.dp]
        self.i_list = [self.i]
        self.t_list = [self.t]
        self.v_list = [self.v]
        
    def derivative(self, t, y):
        p, dp, x, dx, i = y
        
        co = math.cos(p)
        si = math.sin(p)
        D = self.E - self.F * (co ** 2)
        
        return [dp, 
                (self.Ap * si - co * (self.Bp * i - self.Cp * dx + self.Dp * si * (dp ** 2))) / D,
                dx,
                (-self.Ax * si * co + self.Bx * i - self.Cx * dx + self.Dx * si * (dp ** 2)) / D,
                self.Ai * self.v - self.Bi * i - self.Ci * dx]
    
    def derivative_L0(self, t, y):
        p, dp, x, dx = y
        
        i = (self.v - self.Ke * dx / self.r) / self.R
        co = math.cos(p)
        si = math.sin(p)
        D = self.E - self.F * (co ** 2)
        
        return [dp, 
                (self.Ap * si - co * (self.Bp * i - self.Cp * dx + self.Dp * si * (dp ** 2))) / D,
                dx,
                (-self.Ax * si * co + self.Bx * i - self.Cx * dx + self.Dx * si * (dp ** 2)) / D]
        
    def iteration_odeint(self):
        t = np.linspace(self.t, self.t + self.Ts, 2)
        sol = odeint(self.derivative_L0, [self.p, self.dp, self.x, self.dx], t, tfirst=True, rtol=self.rtol, atol=self.atol)
    
        self.t = t[-1]
        self.p = sol[-1, 0]
        self.dp = sol[-1, 1]
        self.x = sol[-1, 2]
        self.dx = sol[-1, 3]
        
    def iteration_odeint_n(self):
        t = np.linspace(self.t, self.t + self.Ts, self.n + 1)
        sol = odeint(self.derivative_L0, [self.p, self.dp, self.x, self.dx], t, tfirst=True, rtol=self.rtol, atol=self.atol)
    
        self.t = t[-1]
        self.p = sol[-1, 0]
        self.dp = sol[-1, 1]
        self.x = sol[-1, 2]
        self.dx = sol[-1, 3]
        self.i = (self.v - self.Ke * self.dx / self.r) / self.R
        
        self.t_list += list(t[1:])
        self.p_list += list(sol[1:, 0])
        self.dp_list += list(sol[1:, 1])
        self.x_list += list(sol[1:, 2])
        self.dx_list += list(sol[1:, 3])
        self.i_list += list((self.v - self.Ke * sol[1:, 3] / self.r) / self.R)
        self.v_list += list(np.ones(self.n) * self.v)
        
    def iteration_odeint_L(self):
        t = np.linspace(self.t, self.t + self.Ts, 2)
        sol = odeint(self.derivative, [self.p, self.dp, self.x, self.dx, self.i], t, tfirst=True, rtol=self.rtol, atol=self.atol)
    
        self.t = t[-1]
        self.p = sol[-1, 0]
        self.dp = sol[-1, 1]
        self.x = sol[-1, 2]
        self.dx = sol[-1, 3]
        self.i = sol[-1, 4]
        
    def iteration_odeint_L_n(self):
        t = np.linspace(self.t, self.t + self.Ts, self.n + 1)
        sol = odeint(self.derivative, [self.p, self.dp, self.x, self.dx, self.i], t, tfirst=True, rtol=self.rtol, atol=self.atol)
    
        self.t = t[-1]
        self.p = sol[-1, 0]
        self.dp = sol[-1, 1]
        self.x = sol[-1, 2]
        self.dx = sol[-1, 3]
        self.i = sol[-1, 4]
        
        self.t_list += list(t[1:])
        self.p_list += list(sol[1:, 0])
        self.dp_list += list(sol[1:, 1])
        self.x_list += list(sol[1:, 2])
        self.dx_list += list(sol[1:, 3])
        self.i_list += list(sol[1:, 4])
        self.v_list += list(np.ones(self.n) * self.v)

    def save_state(self):
        self.t_list.append(self.t)
        self.p_list.append(self.p)
        self.dp_list.append(self.dp)
        self.x_list.append(self.x)
        self.dx_list.append(self.dx)
        self.i_list.append(self.i)
        self.v_list.append(self.v)
    
    def get_x(self):
        return np.array(self.x_list)
    
    def get_dx(self):
        return np.array(self.dx_list)
    
    def get_p_180(self):
        return np.array(self.p_list) * 180.0 / math.pi
    
    def get_p_360(self):
        val = np.array(self.p_list) * 180.0 / math.pi
        for j in range(len(self.p_list)):
            if val[j] < 0:
                val[j] += 360.0
                
        return val
    
    def get_dp(self):
        return np.array(self.dp_list) * 180.0 / math.pi
    
    def get_i(self):
        return np.array(self.i_list)
    
    def get_t(self):
        return np.array(self.t_list)
    
    def get_v(self):
        return np.array(self.v_list)
    
    def is_done(self):
        return self.p > self.p_lim or self.p < -self.p_lim or self.x > self.x_lim or self.x < -self.x_lim
        
    def get_state(self):
        return np.array([self.x, self.dx, self.p, self.dp])
    
    def get_reward(self, x, p):
        return -self.lambda_p * np.abs(p) - self.lambda_x * np.abs(x)
    
    