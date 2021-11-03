import os

# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

import gym
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from actor_critic_networks import Critic, Policy
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# T is a global counter
# Tmax is total steps overall
# t is the local counter per process

class ActorCriticWorker(mp.Process):
    def __init__(self,env_name,global_critic,global_actor,opt,T,lock,global_t_max, summary_writer = None,
                 eval_runs = 10, gamma = 0.99,
                 max_step=100,
                 beta=0.01):
        super(ActorCriticWorker, self).__init__()
        self.env = gym.make(env_name)
        self.t = 0
        self.t_max = max_step  # max steps for episode/rollout
        self.T = T
        self.lock = lock
        self.gamma = gamma
        self.opt = opt
        self.global_t_max = global_t_max
        self.beta = beta

        self.actor = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic = Critic(self.env.observation_space.shape[0])
        self.global_critic = global_critic
        self.global_actor = global_actor

        # TFBoard settings
        self.TFB_Counter = 0
        # self.summary_writer = summary_writer
        self.eval_counter = 0
        self.eval_runs = eval_runs

        log_dir = 'logs'+self.name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.summary_writer = SummaryWriter(log_dir=log_dir)


    def run(self):

        while self.T.value < self.global_t_max:

            # 1. Sync local from global - we need this for the actor: get_action()
            self.actor.load_state_dict(self.global_actor.state_dict())
            self.critic.load_state_dict(self.global_critic.state_dict())

            # Tensorboard - run evaluation for n episodes and collect the states to TFBoard
            # if self.summary_writer is not None and (self.T.value - self.TFB_Counter) > 500:
            if self.name == 'ActorCriticWorker-1' and (self.T.value - self.TFB_Counter) > 50:
                print("Tensorboard is active")
                self.TFB_Counter = self.T.value
                eval_reward = self.eval()
                self.summary_writer.add_scalar("eval_reward", eval_reward, self.eval_counter)
                self.eval_counter +=1

            # 2. Create a rollout
            t_start = self.t
            state   = self.env.reset() #giving us a state from the gym env.
            done    = False
            states  = []
            actions = []
            rewards = []
            returns = []

            while not done and (self.t - t_start+1)%self.t_max !=0:
                action = self.actor.get_action(torch.tensor(state, dtype=torch.float).reshape(1,-1))
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
            entropy_loss = -1*self.beta * self.actor.entropy(states_t) # n_batch x 1
            # Take mean of the actor and critic loss
            total_loss = (critic_loss + actor_loss + entropy_loss).mean()

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

    def eval(self):

        for _ in range(self.eval_runs):
            state = self.env.reset()
            done = False
            eval_t = 0
            accumulated_reward = 0.0

            while not done and eval_t <= self.t_max:
                action = self.actor.get_action(torch.tensor(state, dtype=torch.float).reshape(1,-1))
                next_state, reward,done, _info = self.env.step(action)
                accumulated_reward += reward
                eval_t +=1
                state = next_state

        return accumulated_reward/self.eval_runs

