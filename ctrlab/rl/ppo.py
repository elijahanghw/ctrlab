import os
from time import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from random import sample

class PPO:
    def __init__(self, 
                 env, 
                 hidden_dim=64, 
                 num_layers=2, 
                 activation='relu',
                 n_steps=1000, 
                 batch_size=256, 
                 n_epochs=10, 
                 gamma=0.999, 
                 lr=3e-4,
                 policy_clip=0.1,
                 gae_lambda=0.95,
                 directory='tmp/ppo'):
        
        self.env = env
        self.num_envs = env.num_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(self.obs_dim, self.action_dim, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation, lr=lr, directory=directory)
        self.critic = Critic(self.obs_dim, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation, lr=lr, directory=directory)
        
    def rollout(self):
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        terminated_buffer = []
        value_buffer = []
        log_prob_buffer = []
        
        rollout_steps = 0
        total_rollout_steps = self.n_steps * self.num_envs
        
        obs = self.env.reset(np.ones(self.num_envs, dtype=bool))[0]
        obs = torch.tensor(obs, dtype=torch.float32)
        
        while rollout_steps < total_rollout_steps:
            # Get action
            mean, std = self.actor(obs)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            actions_clipped = torch.clamp(actions, -1, 1)
            log_probs = dist.log_prob(actions_clipped)
            values = self.critic(obs).squeeze(-1)
            
            # Step environment
            obs_new, reward, terminated, truncated, _ = self.env.step(actions_clipped.numpy())
            dones = terminated | truncated

            # Collect data
            obs_buffer.append(obs)
            action_buffer.append(actions.detach())
            reward_buffer.append(torch.tensor(reward, dtype=torch.float32))
            terminated_buffer.append(torch.tensor(terminated, dtype=torch.float32))
            value_buffer.append(values.detach())
            log_prob_buffer.append(log_probs.detach())
            
            obs, _ = self.env.reset(dones)
            obs = torch.tensor(obs, dtype=torch.float32)
            
            rollout_steps += self.num_envs
            
        obs_buffer = torch.stack(obs_buffer)
        action_buffer = torch.stack(action_buffer)
        reward_buffer = torch.stack(reward_buffer)
        terminated_buffer = torch.stack(terminated_buffer)
        value_buffer = torch.stack(value_buffer)
        log_prob_buffer = torch.stack(log_prob_buffer)
        
        # Final critic values to bootstrap GAE
        last_values = self.critic(obs).squeeze(-1)
        
        return obs_buffer, action_buffer, reward_buffer, terminated_buffer, value_buffer, log_prob_buffer, last_values, rollout_steps
    
    def learn(self, total_timesteps, save_interval=5000000):
        print("Learning...")
        start_time = time()
        T = 0
        save_T = 0
        while T < total_timesteps:
            obs_buffer, action_buffer, reward_buffer, terminated_buffer, value_buffer, log_prob_buffer, last_values, rollout_steps = self.rollout()
            # Compute advantages
            advantages, returns = self.compute_advantages(reward_buffer, value_buffer, terminated_buffer, last_values)

            self.update_policy(obs_buffer, action_buffer, log_prob_buffer, advantages, returns)
                
            T += rollout_steps
            save_T += rollout_steps
            
            print(f"Total timesteps: {T}/{total_timesteps}, Time: {time() - start_time:.2f}s")
            if save_T >= save_interval:
                self.save_models()
                save_T = 0
                print(f"Models saved at timestep {T}")
            
    def compute_advantages(self, rewards, values, dones, last_values):
        T, N = rewards.shape  # T = rollout steps, N = num_envs
        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(N)
        values = values.detach()
        last_values = last_values.detach()
        
        for t in reversed(range(T)):
            next_value = last_values if t == T - 1 else values[t + 1]
            mask = 1.0 - dones[t] # reset advantage if done
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * mask * last_adv
            advantages[t] = last_adv

        returns = advantages + values
        return advantages.flatten(), returns.flatten()
    
    
    def update_policy(self, obs_buffer, action_buffer, log_prob_buffer, advantages, returns):
        T, N, obs_dim = obs_buffer.shape
        _, _, act_dim = action_buffer.shape

        obs = obs_buffer.view(T * N, obs_dim)
        actions = action_buffer.view(T * N, act_dim)
        old_log_probs = log_prob_buffer.view(T * N, act_dim).sum(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = T*N
        minibatch_size = self.batch_size

        for _ in range(self.n_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Compute new log probs and entropy
                mean, std = self.actor(mb_obs)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1)

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                # Critic loss
                values = self.critic(mb_obs).squeeze(-1)
                critic_loss = F.mse_loss(values, mb_returns)

                # Update actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # Update critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
    
    def save_models(self):
        print("Saving models...")
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        if not os.path.exists('tmp/ppo'):
            os.makedirs('tmp/ppo')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        print("Loading models...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    
    
class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, num_layers=2, activation='relu', lr=3e-4, directory='tmp/ppo'):
        super(Actor, self).__init__()

        self.checkpoint_file = os.path.join(directory, 'actor.pth')
        
        self.fcin = nn.Linear(in_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.mu = nn.Linear(hidden_dim, out_dim)
        
        self.log_std = nn.Parameter(torch.zeros(1, out_dim))
            
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        x = self.activation(self.fcin(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        mean = self.mu(x)
        
        std = self.log_std.exp()
        
        return mean, std
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, activation='relu', lr=3e-4, directory='tmp/ppo'):
        super(Critic, self).__init__()
        
        self.checkpoint_file = os.path.join(directory, 'critic.pth')

        self.fcin = nn.Linear(in_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.fcout = nn.Linear(hidden_dim, 1)
            
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        x = self.activation(self.fcin(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        value = self.fcout(x)
        
        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        
