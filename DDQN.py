import numpy as np
import torch as T

from model import Model
from neural_networks import NN
    
class Agent(object):
    def __init__(self, device, gamma=0.99, lr=3e-6, max_v=12.0, n_actions=9, model=Model(), n_states=4, 
                 eps_start=1.0, eps_stop=0.01, eps_dec=1e-3, buffer_size=19e3, batch_size=128, att_target_period=400, 
                 internal_layers_size=[3000, 2000, 1000, 500, 250]):
        
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.actions_v = np.linspace(-max_v, max_v, n_actions)
        self.actions = np.arange(n_actions)
        self.n_actions = n_actions
        self.n_states = n_states
        self.model = model
        self.eps = eps_start
        self.eps_stop = eps_stop
        self.eps_dec = eps_dec
        self.step = 0
        self.att_target_period = att_target_period
        self.loss_lp = 0.0
        self.loss_f = 0.99
        final_reward = model.get_reward(model.x_lim, model.p_lim)
        self.final_reward_gamma = final_reward / (1 - gamma)
        
        self.buffer_size = np.int64(buffer_size)
        self.buffer_counter = 0
        self.batch_size = batch_size
        self.buffer_state = np.zeros((self.buffer_size, n_states), dtype=np.float32)
        self.buffer_state_ = np.zeros((self.buffer_size, n_states), dtype=np.float32)
        self.buffer_action = np.zeros(self.buffer_size, dtype=np.int32)
        self.buffer_reward = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_done = np.zeros(self.buffer_size, dtype=bool) 
        
        self.step_hist, self.eps_hist, self.score_hist, self.t_max_hist, self.loss_hist = [], [], [], [], []
        self.run_hist, self.t_hist, self.x_hist, self.dx_hist, self.p_hist, self.dp_hist, self.v_hist = [], [], [], [], [], [], []
        
        self.q_eval = NN(device=device, lr=lr, layers_size=[n_states] + internal_layers_size + [self.n_actions])
        self.q_targ = NN(device=device, lr=lr, layers_size=[n_states] + internal_layers_size + [self.n_actions])
    
    def best_action(self, state):
        state = T.tensor(state, dtype=T.float32).to(self.device)
        action = self.q_eval.forward(state)
        return T.argmax(action).item()
    
    def best_action_v(self, state):
        return self.actions_v[self.best_action(state)]
    
    def get_action(self, state):
        if np.random.random() > self.eps:
            return self.best_action(state)
        
        else:
            return np.random.choice(self.actions)
        
    def get_action_v(self, state):
        return self.actions_v[self.get_action(state)]
        
    def buffer_save(self, state, action, reward, state_, done):
        index = self.buffer_counter % self.buffer_size
        self.buffer_state[index] = state
        self.buffer_state_[index] = state_
        self.buffer_action[index] = np.where(self.actions_v == action)[0]
        self.buffer_reward[index] = reward
        self.buffer_done[index] = done
        self.buffer_counter += 1

    def buffer_load(self):
        max_index = min(self.buffer_counter, self.buffer_size)
        batch = np.random.choice(max_index, self.batch_size, replace=False)
        states = T.tensor(self.buffer_state[batch]).to(self.device)
        states_ = T.tensor(self.buffer_state_[batch]).to(self.device)
        actions = T.tensor(self.buffer_action[batch], dtype=T.long).to(self.device)
        rewards = T.tensor(self.buffer_reward[batch]).to(self.device)
        dones = T.tensor(self.buffer_done[batch]).to(self.device)
        return states, states_, actions, rewards, dones
    
    def save_run(self, run):
        self.run_hist.append(run)
        self.t_hist.append(self.model.get_t())
        self.x_hist.append(self.model.get_x())
        self.dx_hist.append(self.model.get_dx())
        self.p_hist.append(self.model.get_p_180())
        self.dp_hist.append(self.model.get_dp())
        self.v_hist.append(self.model.get_v())
        
    def save_step(self, score):
        self.step_hist.append(self.step)
        self.eps_hist.append(self.eps)
        self.score_hist.append(score)
        self.t_max_hist.append(self.model.t)
        self.loss_hist.append(self.loss_lp)

    def learn(self):
        if self.buffer_counter < self.batch_size:
            return

        self.step += 1
        if self.step % self.att_target_period == 0:
            self.q_targ.load_state_dict(self.q_eval.state_dict())

        self.q_eval.optimizer.zero_grad()
        states, states_, actions, rewards, dones = self.buffer_load()
        batch_indexes = np.arange(self.batch_size)

        q_eval = self.q_eval.forward(states)[batch_indexes, actions]
        q_eval_ = self.q_eval.forward(states_)
        actions_ = T.argmax(q_eval_, dim=1)
        q_targ = self.q_targ.forward(states_)
        q_targ = q_targ[batch_indexes, actions_]
        q_targ[dones] = self.final_reward_gamma
        q_eval_update = rewards + self.gamma*q_targ

        loss = self.q_eval.loss(q_eval_update, q_eval).to(self.device)
        loss.backward()
        self.q_eval.optimizer.step()
        
        self.loss_lp = self.loss_f * self.loss_lp + (1 - self.loss_f) * loss.item()
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_stop else self.eps_stop





