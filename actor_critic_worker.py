import os

# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

import gym
import torch
import numpy as np
import torch.multiprocessing as mp
from actor_critic_networks import Critic, Policy
from torch.utils.tensorboard import SummaryWriter
# import wandb

# T is a global counter
# Tmax is total steps overall
# t is the local counter per process
# max_step: max steps per episode

class ActorCriticWorker(mp.Process):
    def __init__(self,env_name,global_critic,global_actor,opt,T,lock,global_t_max,
                 eval_runs = 10, gamma = 0.99,
                 max_step=1000,
                 beta=0.01,
                 DEBUG_F = False,
                 eval_freq = 5000):
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
        self.DEBUG_F = DEBUG_F
        self.actor = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic = Critic(self.env.observation_space.shape[0])
        self.global_critic = global_critic
        self.global_actor = global_actor
        self.eval_freq = eval_freq

        # TFBoard settings
        self.TFB_Counter = 0  # keeps track of how often we want to log to tensorboard
        self.setup_logging(tensorboard=True, wandb=False)

        # Eval run settings
        self.eval_counter = 0
        self.eval_runs = eval_runs

    def setup_logging(self, tensorboard, wandb):
        if tensorboard:
            log_dir = 'logs/'+self.name
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        # if wandb:
        #     self.run = wandb.init(project="mlc-a3c", entity="moodlep", group="a3c")

    def run(self):
        self.setup_logging(tensorboard=True, wandb=False)

        while self.T.value < self.global_t_max:
            if self.DEBUG_F: print("Outer Loop: ", self.T.value)
            # 1. Sync local from global - we need this for the actor: get_action()
            self.actor.load_state_dict(self.global_actor.state_dict())
            self.critic.load_state_dict(self.global_critic.state_dict())

            # Tensorboard - run evaluation for n episodes and collect the stats to TFBoard
            if (self.T.value - self.TFB_Counter) > self.eval_freq:
                if self.DEBUG_F: print("Inside tensorboard if statement: ", (self.T.value - self.TFB_Counter))
                self.TFB_Counter = self.T.value
                eval_reward = self.eval()
                if self.DEBUG_F: print("Before SummaryWriter.add_scalar ")
                self.summary_writer.add_scalar("eval_reward", eval_reward, self.T.value)
                # wandb.log({"eval_reward": eval_reward, "T": self.T.value})
                if self.DEBUG_F: print("After SummaryWriter.add_scalar")
                # self.summary_writer.flush()
                self.eval_counter +=1

            # 2. Create a rollout
            t_start = self.t
            state   = self.env.reset() #giving us a state from the gym env.
            done    = False
            states  = []
            actions = []
            rewards = []
            returns = []

            if self.DEBUG_F: print("starting rollout: ", t_start)

            while not done and (self.t - t_start+1)%self.t_max !=0:
                if self.DEBUG_F: print("Inner Loop: ", self.t)

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
            if self.DEBUG_F: print("Calculate Reward: ", R)


            for i in range(len(states)-1,-1,-1):  #Reverse because this is a bellman-type of calculation (you know all your rewards from t to the end)
                R = rewards[i] + self.gamma*R
                returns.append(R)
                if self.DEBUG_F: print("Returns Calc For Loop: ", i)
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
            if self.DEBUG_F: print("Total Loss at ", self.T.value, " is ", total_loss.item())
            # wandb.log({"loss": total_loss.item()})

            # 4. Calculate grad and update optimiser
            self.opt.zero_grad()
            total_loss.backward()
            if self.DEBUG_F: print("Calc Grads Backward: ")

            # align global grads to local grads
            for gp, lp in zip(self.global_critic.parameters(), self.critic.parameters()):
                gp._grad = lp.grad
            for gp, lp in zip(self.global_actor.parameters(), self.actor.parameters()):
                gp._grad = lp.grad
            if self.DEBUG_F: print("Global Grads Updated: ")

            # take a step!
            self.opt.step()
            if self.DEBUG_F: print("Optimiser Step: ")

        self.summary_writer.close()

    def eval(self):
        print("timer:",self.T.value)
        accumulated_reward = 0.0
        for i in range(self.eval_runs):
            state = self.env.reset()
            done = False
            eval_t = 0
            if self.DEBUG_F: print("Eval For Loop: ", i)

            while not done and eval_t <= self.t_max:
                action = self.actor.get_action(torch.tensor(state, dtype=torch.float).reshape(1,-1))
                next_state, reward,done, _info = self.env.step(action)
                accumulated_reward += reward
                eval_t +=1
                state = next_state
                if self.DEBUG_F: print("Eval Loop: ", accumulated_reward, eval_t)

        return accumulated_reward/self.eval_runs

