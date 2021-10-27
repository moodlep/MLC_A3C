import os

# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

import gym
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from actor_critic_networks import Critic, Policy

# T is a global counter
# Tmax is total steps overall
# t is the local counter per process

class ActorCriticWorker(mp.Process):
    def __init__(self,env_name,global_critic,global_actor,opt,T,lock,global_t_max, gamma = 0.99,max_step=100):
        super(ActorCriticWorker, self).__init__()
        self.env = gym.make(env_name)
        self.t = 0
        self.max_step = max_step  # max steps for episode/rollout
        self.T = T
        self.lock = lock
        self.gamma = gamma
        self.opt = opt
        self.global_t_max = global_t_max

        self.actor = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic = Critic(self.env.observation_space.shape[0])
        self.global_critic = global_critic
        self.global_actor = global_actor

    def run(self):

        # 1. Sync local from global - we need this for the actor: get_action()
        self.actor.load_state_dict(self.global_actor.state_dict())
        self.critic.load_state_dict(self.global_critic.state_dict())

        # 2. Create a rollout
        t_start = self.t
        state   = self.env.reset() #giving us a state from the gym env.
        done    = False
        states  = []
        actions = []
        rewards = []
        returns = []

        while not done and (self.t - t_start+1)%self.max_step !=0:
            action = self.actor.get_action(torch.tensor(state, dtype=torch.float).reshape(1,-1))
            #print(action)
            next_state, reward,done, _info = self.env.step(action)
            rewards.append(reward)
            actions.append(action)
            states.append(state)
            state = next_state
            self.t  += 1
            # lock memory
            with self.lock:
                self.T.value +=1

        # Calculate reward
        with torch.no_grad():
            if not done:
                R = self.critic(torch.tensor(state,dtype = torch.float)).item() #calculating the value function
            else:
                R = 0.0

        for i in range(len(states)-1,-1,-1):  #Reverse because this is a bellman-type of calculation (you know all your rewards from t to the end)
            R = rewards[i] + self.gamma*R
            returns.append(R)
        returns.reverse() # list of returns

        # 3. Calculating Loss
        states_t = torch.tensor(np.array(states), dtype = torch.float)
        actions_t = torch.tensor(actions, dtype = torch.int)
        returns_t = torch.tensor(returns, dtype = torch.float)

        td_error = returns_t - self.critic(states_t)	# n_batch x 1
        critic_loss = (td_error)**2 # 1 x 1
        actor_loss = -1.0*td_error.detach()*self.actor.log_prob(states_t, actions_t) # n_batch x 1
        # Take mean of the actor and critic loss
        total_loss = (critic_loss + actor_loss).mean()

        # 4. Calculate grad and update optimiser
        self.opt.zero_grad()
        total_loss.backward()

        # align global grads to local grads
        for gp, lp in zip(self.global_critic.parameters(), self.critic.parameters()):
            gp._grad = lp.grad
        for gp, lp in zip(self.global_actor.parameters(), self.actor.parameters()):
            gp._grad = lp.grad

        # take a step!
        self.opt.step()

