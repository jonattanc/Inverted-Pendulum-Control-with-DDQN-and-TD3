import numpy as np
import torch as T
import torch.nn.functional as F

from model import Model
from neural_networks import NN
    
class Agent(object):
    def __init__(self, device, gamma=0.99, lr=5e-5, model=Model(), n_states=4, n_actions=1, max_action=12.0, action_norm=0.02, 
                 buffer_size=1e2, batch_size=128, tau=0.005, noise_std_a=0.01, noise_std_a_=0.12, noise_clamp_factor=5.0, 
                 update_actor_interval=2, random_steps=3e3, internal_layers_size=[2000, 1000, 500, 250]):
        
        self.device = device
        self.gamma = gamma
        self.lr_actor = lr
        self.lr_crit = lr
        self.n_states = n_states
        self.model = model
        self.step = 0
        self.noise_std_a = noise_std_a
        self.noise_std_a_ = noise_std_a_
        self.noise_clamp_factor = noise_clamp_factor
        self.random_steps = random_steps
        self.max_action = max_action
        self.update_actor_iter = update_actor_interval
        self.loss_actor_lp = 0.0
        self.loss_crit1_lp = 0.0
        self.loss_crit2_lp = 0.0
        self.loss_f = 0.99
        final_reward = model.get_reward(model.x_lim, model.p_lim)
        self.final_reward_gamma = final_reward / (1 - gamma)
        self.action_norm = action_norm
        
        self.buffer_size = np.int64(buffer_size)
        self.buffer_counter = 0
        self.batch_size = batch_size
        self.buffer_state = np.zeros((self.buffer_size, n_states), dtype=np.float32)
        self.buffer_state_ = np.zeros((self.buffer_size, n_states), dtype=np.float32)
        self.buffer_action = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_reward = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_done = np.zeros(self.buffer_size, dtype=bool) 
        
        self.step_hist, self.score_hist, self.t_max_hist, self.loss_hist, self.loss_crit1_hist, self.loss_crit2_hist = [], [], [], [], [], []
        self.run_hist, self.t_hist, self.x_hist, self.dx_hist, self.p_hist, self.dp_hist, self.v_hist = [], [], [], [], [], [], []
        
        self.actor_eval = NN(device=device, lr=self.lr_actor, layers_size=[n_states] + internal_layers_size + [n_actions])
        self.actor_targ = NN(device=device, lr=self.lr_actor, layers_size=[n_states] + internal_layers_size + [n_actions])
        self.crit1_eval = NN(device=device, lr=self.lr_crit, layers_size=[n_states + n_actions] + internal_layers_size + [1])
        self.crit1_targ = NN(device=device, lr=self.lr_crit, layers_size=[n_states + n_actions] + internal_layers_size + [1])
        self.crit2_eval = NN(device=device, lr=self.lr_crit, layers_size=[n_states + n_actions] + internal_layers_size + [1])
        self.crit2_targ = NN(device=device, lr=self.lr_crit, layers_size=[n_states + n_actions] + internal_layers_size + [1])

        self.tau = 1
        self.update_NN()
        self.tau = tau
        
    def best_action_v(self, state):
        state = T.tensor(state, dtype=T.float32).to(self.device)
        action = self.actor_eval.forward(state).to(self.device)
        action = T.tanh(action) * self.max_action
        return action.item()
    
    def get_action_v(self, state):
        if self.step < self.random_steps:
            action = T.tensor(np.random.uniform(-self.max_action, self.max_action)).to(self.device)
            
        else:
            state = T.tensor(state, dtype=T.float32).to(self.device)
            action = self.actor_eval.forward(state).to(self.device)
            action = T.tanh(action) * self.max_action
            
        action_noise = action + T.tensor(np.random.normal(scale=self.noise_std_a), dtype=T.float32).to(self.device) * self.max_action
        action_noise = T.clamp(action_noise, - self.max_action, self.max_action)
        
        return action_noise.item()
        
    def buffer_save(self, state, action, reward, state_, done):
        index = self.buffer_counter % self.buffer_size
        self.buffer_state[index] = state
        self.buffer_state_[index] = state_
        self.buffer_action[index] = action
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
        self.score_hist.append(score)
        self.t_max_hist.append(self.model.t)
        self.loss_hist.append(self.loss_actor_lp)
        self.loss_crit1_hist.append(self.loss_crit1_lp)
        self.loss_crit2_hist.append(self.loss_crit2_lp)
        
    def update_NN(self):
        actor_eval_param = dict(self.actor_eval.named_parameters())
        actor_targ_param = dict(self.actor_targ.named_parameters())
        crit1_eval_param = dict(self.crit1_eval.named_parameters())
        crit1_targ_param = dict(self.crit1_targ.named_parameters())
        crit2_eval_param = dict(self.crit2_eval.named_parameters())
        crit2_targ_param = dict(self.crit2_targ.named_parameters())

        for i in actor_targ_param:
            actor_targ_param[i] = self.tau * actor_eval_param[i].clone() + (1 - self.tau) * actor_targ_param[i].clone()
            
        for i in crit1_targ_param:
            crit1_targ_param[i] = self.tau * crit1_eval_param[i].clone() + (1 - self.tau) * crit1_targ_param[i].clone()
            
        for i in crit2_targ_param:
            crit2_targ_param[i] = self.tau * crit2_eval_param[i].clone() + (1 - self.tau) * crit2_targ_param[i].clone()

        self.actor_targ.load_state_dict(actor_targ_param)
        self.crit1_targ.load_state_dict(crit1_targ_param)
        self.crit2_targ.load_state_dict(crit2_targ_param)

    def learn(self):
        if self.buffer_counter < self.batch_size:
            return

        self.step += 1

        states, states_, actions, rewards, dones = self.buffer_load()
        actions_ = T.tanh(self.actor_targ.forward(states_)) * self.max_action
        noise_ = T.tensor(np.random.normal(scale=self.noise_std_a_, size=self.batch_size), dtype=T.float32).to(self.device)
        noise_ = T.clamp(noise_, -self.noise_clamp_factor * self.noise_std_a_, self.noise_clamp_factor * self.noise_std_a_).unsqueeze(1)
        actions_noise_ = actions_ + noise_ * self.max_action
        actions_noise_ = T.clamp(actions_noise_, -self.max_action, self.max_action)
        
        actions_noise_states_ = T.cat([actions_noise_ * self.action_norm, states_], dim=1)
        actions_states = T.cat([actions.unsqueeze(1) * self.action_norm, states], dim=1)
        q1_ = self.crit1_targ.forward(actions_noise_states_)
        q2_ = self.crit2_targ.forward(actions_noise_states_)
        q1 = self.crit1_eval.forward(actions_states)
        q2 = self.crit2_eval.forward(actions_states)
        
        rewards = rewards.unsqueeze(1)
        q1_[dones] = self.final_reward_gamma
        q2_[dones] = self.final_reward_gamma
        q_target = rewards + self.gamma * T.min(q1_, q2_)

        self.crit1_eval.optimizer.zero_grad()
        self.crit2_eval.optimizer.zero_grad()
        q1_loss = F.mse_loss(q_target, q1)
        q2_loss = F.mse_loss(q_target, q2)
        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.crit1_eval.optimizer.step()
        self.crit2_eval.optimizer.step()

        if self.step % self.update_actor_iter == 0:
            self.actor_eval.optimizer.zero_grad()
            actor_actions = T.tanh(self.actor_eval.forward(states)) * self.max_action
            actor_actions_states = T.cat([actor_actions * self.action_norm, states], dim=1)
            actor_loss = self.crit1_eval.forward(actor_actions_states)
            actor_loss = -T.mean(actor_loss)
            actor_loss.backward()
            self.actor_eval.optimizer.step()
            self.update_NN()

            self.loss_actor_lp = self.loss_f * self.loss_actor_lp + (1 - self.loss_f) * actor_loss.item()
            self.loss_crit1_lp = self.loss_f * self.loss_crit1_lp + (1 - self.loss_f) * q1_loss.item()
            self.loss_crit2_lp = self.loss_f * self.loss_crit2_lp + (1 - self.loss_f) * q2_loss.item()





