import numpy as np
import math
import ipywidgets
import random
import copy
from IPython.display import display

def train_agents(agent, max_steps=1e3, save_run_period=500, sim_time=5.0, x0=0.2, dx0=0.0, p0=14.0, dp0=0.0, 
                 n_agt=2, find_best_agent=False, get_history=False, score_limit=80, eval_interval=30, eval_runs=100):
    
    print(f'training with {int(max_steps)} max_steps')
    
    if get_history or find_best_agent:
        print(f'running {int(eval_runs)} episodes to evaluate agents every {int(eval_interval)} training episodes')
    
    intProg = ipywidgets.IntProgress(min=0, max=n_agt)
    lbl = ipywidgets.Label(value="0/{:,.0f}".format(n_agt))
    box = ipywidgets.HBox([intProg, lbl])
    display(box)
    results = []
    
    for i in range(n_agt):
        copy_agent = copy.deepcopy(agent)
        last_agent, best_agent, history = train_single_agent(agent=copy_agent, max_steps=max_steps, save_run_period=save_run_period, 
                                                             sim_time=sim_time, x0=x0, dx0=dx0, p0=p0, dp0=dp0, find_best_agent=find_best_agent, 
                                                             get_history=get_history, score_limit=score_limit, eval_interval=eval_interval, 
                                                             eval_runs=eval_runs)
        results.append([last_agent, best_agent, history])
        lbl.value = "{:,.0f}/{:,.0f}".format(i + 1, n_agt)
        intProg.value = i + 1
        
    return results

def train_single_agent(agent, max_steps=1000, save_run_period=500, sim_time=5.0, x0=0.2, dx0=0.0, p0=14.0, dp0=0.0, 
                       find_best_agent=False, get_history=False, score_limit=80, eval_interval=30, eval_runs=100):
    
    final_reward = agent.model.get_reward(agent.model.x_lim, agent.model.p_lim)
    
    iterations_per_episode = int(sim_time / agent.model.Ts)
    intProg = ipywidgets.IntProgress(min=0, max=max_steps)
    lbl = ipywidgets.Label(value='')
    box = ipywidgets.HBox([intProg, lbl])
    display(box)
    
    filter_factor = 0.9
    episode_time_filtered = 0.0
    score_filtered = final_reward * iterations_per_episode
    run = 0
    
    if find_best_agent:
        best_agent = copy.deepcopy(agent)
        best_score = -score_limit
        
    if get_history:
        run_list = []
        score_mean_list = []
        episode_lengths_mean_list = []
    
    while agent.step < max_steps:
        run += 1
        x = random.uniform(-x0, x0)
        dx = random.uniform(-dx0, dx0)
        p = random.uniform(-p0, p0) * math.pi / 180.0
        dp = random.uniform(-dp0, dp0) * math.pi / 180.0
        agent.model.reset_var(x=x, dx=dx, p=p, dp=dp)
        score = 0.0
        
        score = run_episode(iterations_per_episode, agent, run, save_run_period, score)

        if run % save_run_period == 0:
            agent.save_run(run)
            
        agent.save_step(score)
        
        episode_time_filtered = episode_time_filtered * filter_factor + agent.model.t * (1 - filter_factor)
        score_filtered = score_filtered * filter_factor + score * (1 - filter_factor)
        lbl.value = "{:.1%}, step: {:,.0f}/{:,.0f}, run: {:,}, t: {:.2f}, score: {:.2f}".format(agent.step / max_steps, agent.step, max_steps, 
                                                                                                run, episode_time_filtered, score_filtered)
        intProg.value = agent.step
        
        if run % eval_interval == 0 and (get_history or (find_best_agent and score_filtered > -score_limit)):
            scores_mean, episode_lengths_mean = evaluate_agent(eval_runs, x0, dx0, p0, dp0, agent, iterations_per_episode)
            
            if get_history:
                run_list.append(run)
                score_mean_list.append(scores_mean)
                episode_lengths_mean_list.append(episode_lengths_mean)
                
            if find_best_agent and scores_mean > best_score:
                best_score = scores_mean
                best_agent = copy.deepcopy(agent)

    if get_history:
        history = [run_list, score_mean_list, episode_lengths_mean_list]
        
    else:
        history = None
        
    if find_best_agent:
        print('best agent score:', best_score)
        
    else:
        best_agent = None
    
    return agent, best_agent, history
    
def run_episode(iterations_per_episode, agent, run, save_run_period, score):
    for i in range(iterations_per_episode):
        state = agent.model.get_state()
        if state[0] < 0:
            state = -state
            sgn = -1
            
        else:
            sgn = 1
            
        action = agent.get_action_v(state)
        agent.model.v = sgn * action
        agent.model.iteration_odeint()
        state_ = agent.model.get_state()
        
        if state_[0] < 0:
            state_ = -state_
            
        done = agent.model.is_done()
        reward = agent.model.get_reward(agent.model.x, agent.model.p)
        agent.buffer_save(state, action, reward, state_, done)
        agent.learn()

        if run % save_run_period == 0:
            agent.model.save_state()
            
        if done:
            score += reward * (iterations_per_episode - i)
            break
        
        else:
            score += reward
            
    return score

def evaluate_agent(eval_runs, x0, dx0, p0, dp0, agent, iterations_per_episode):
    scores, episode_lengths = [], []
    
    for j in range(eval_runs):
        x = random.uniform(-x0, x0)
        dx = random.uniform(-dx0, dx0)
        p = random.uniform(-p0, p0) * math.pi / 180.0
        dp = random.uniform(-dp0, dp0) * math.pi / 180.0
        agent.model.reset_var(x=x, dx=dx, p=p, dp=dp)
        score = 0.0
        
        for i in range(iterations_per_episode):
            state = agent.model.get_state()
            if state[0] < 0:
                state = -state
                sgn = -1
                
            else:
                sgn = 1
                
            action = agent.best_action_v(state)
            agent.model.v = sgn * action
            agent.model.iteration_odeint()
            done = agent.model.is_done()
            reward = agent.model.get_reward(agent.model.x, agent.model.p)
            if done:
                score += reward * (iterations_per_episode - i)
                break
            
            else:
                score += reward
                
        scores.append(score)
        episode_lengths.append(agent.model.t)
        
    return np.mean(scores), np.mean(episode_lengths)












