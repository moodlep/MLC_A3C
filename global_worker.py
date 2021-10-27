import os

# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

import gym
import torch.multiprocessing as mp

from actor_critic_networks import Critic, Policy
from shared_adam import SharedAdam
from actor_critic_worker import ActorCriticWorker


if __name__ == '__main__':
    mp.freeze_support()  # https://github.com/pytorch/pytorch/issues/5858

    # Create a dummy env
    env = gym.make("LunarLander-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    global_t_max = 10000

    global_critic = Critic(state_dim)
    global_actor = Policy(state_dim,action_dim)
    global_critic.share_memory()
    global_actor.share_memory()
    env_name = "LunarLander-v2"

    global_opt = SharedAdam(list(list(global_critic.parameters()) + list(global_actor.parameters())))
    global_ctr = mp.Value('i',0)  # T - the global step counter
    lock = mp.Lock()  # send to worker when it needs to update global counter

    pr = [ ActorCriticWorker(env_name,global_critic,global_actor,global_opt,global_ctr,lock, global_t_max) for _ in range(
        mp.cpu_count())]
    # pr = [ ActorCriticWorker(env_name,global_critic,global_actor,global_opt,global_ctr,lock) for _ in range(1)]

    for p in pr:
        print(type(p))
        p.start()  # start each ActorCriticWorker process

    for p in pr:
        p.join()  # wait for process to finish