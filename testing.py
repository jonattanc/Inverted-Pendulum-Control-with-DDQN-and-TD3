import numpy as np
import math
import ipywidgets
import random
from IPython.display import display

from model import Model

def test_agent(agent, size=1, sim_time=2.0, x0=0.195, dx0=0.0, p0=13.5, dp0=0.0):
    it = int(sim_time / agent.model.Ts)
    f1 = ipywidgets.IntProgress(min=0, max=it - 1)
    f2 = ipywidgets.IntProgress(min=0, max=size - 1)
    display(f1)
    display(f2)
    models = []

    for s in range(size):
        x = random.uniform(-x0, x0)
        dx = random.uniform(-dx0, dx0)
        p = random.uniform(-p0, p0) * math.pi / 180.0
        dp = random.uniform(-dp0, dp0) * math.pi / 180.0
        
        model = Model(x=x, dx=dx, p=p, dp=dp)
        models.append(model)
        
        for j in range(it):
            state = model.get_state()
            if state[0] < 0:
                state = -state
                sgn = -1
                
            else:
                sgn = 1
                
            action = agent.best_action_v(state)
            model.v = sgn * action
            model.iteration_odeint_n()
            if model.is_done():
                break
            
            f1.value = j
            
        f2.value = s
        
    return models

def test_agent_with_input(agents_data, sim_time=2.0, x0=0.0, dx0=0.0, p0=0.0, dp0=0.0, input_signal='step', 
                          x1=0.1, t1=0.5, r1=0.02, t2=1.0, use_best_agents=True, 
                          noise_std_x=0.0, noise_std_dx=0.0, noise_std_p=0.0, noise_std_dp=0.0, noise_std_v=0.0):
    
    if use_best_agents:
        agents = [i[1] for i in agents_data]
        
    else:
        agents = [i[0] for i in agents_data]
        
    it = int(sim_time / agents[0].model.Ts)
    it1 = int(t1 / agents[0].model.Ts)
    it2 = int(t2 / agents[0].model.Ts)
    
    models = []
    
    for i in range(len(agents)):
        x = x0
        dx = dx0
        p = p0 * math.pi / 180.0
        dp = dp0 * math.pi / 180.0
        
        model = Model()
        model.reset_var(x=x, dx=dx, p=p, dp=dp)
        
        x_input = []
        y_input = []
        
        for j in range(it):
            state = model.get_state()
            
            state[0] = state[0] + np.random.normal(scale=noise_std_x)
            state[1] = state[1] + np.random.normal(scale=noise_std_dx)
            state[2] = state[2] + np.random.normal(scale=noise_std_p) * math.pi / 180.0
            state[3] = state[3] + np.random.normal(scale=noise_std_dp) * math.pi / 180.0
            
            if input_signal == 'step':
                if j >= it1:
                    state[0] = state[0] - x1
                    y_input.append(x1)
                    
                else:
                    y_input.append(0)
                    
            else:
                if j >= it2:
                    state[0] = state[0] - r1 * (j - it1) * model.Ts + 2 * r1 * (j - it2) * model.Ts
                    y_input.append(r1 * (j - it1) * model.Ts - 2 * r1 * (j - it2) * model.Ts)
                    
                elif j >= it1:
                    state[0] = state[0] - r1 * (j - it1) * model.Ts
                    y_input.append(r1 * (j - it1) * model.Ts)
                    
                else:
                    y_input.append(0)
                    
            x_input.append(j * agents[0].model.Ts)
            
            if state[0] < 0:
                state = -state
                sgn = -1
                
            else:
                sgn = 1
                
            action = agents[i].best_action_v(state) + np.random.normal(scale=noise_std_v)
            model.v = sgn * action
            model.iteration_odeint_n()
            
        models.append(model)
        
    return models, [x_input, y_input, sim_time]









