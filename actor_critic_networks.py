import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

"""
Current format: separate networks for critic and actor
TODO: create a shared architecture version 
"""
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=100):
        super(Policy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, action_dim)

    def forward(self, state):
        print(state)
        q = F.leaky_relu(self.l1(state))
        q = F.leaky_relu(self.l2(q))
        print(q.shape)
        return F.softmax(self.l3(q), dim=1)

    def get_action(self, state):
        # state = torch.tensor(state, dtype=torch.double)
        with torch.no_grad():
            policy = self.forward(state)
            dist = torch.distributions.Categorical(policy)
        return dist.sample().item()  # returns a batch of values

    def log_prob(self, state, actions):
        # Part of the loss term
        pol = self.forward(state)
        log_prob = torch.distributions.Categorical(pol).log_prob(actions)
        return log_prob

    def entropy(self, state):
        pol = self.forward(state)
        return torch.distributions.Categorical(pol).entropy()



class Critic(nn.Module):
    def __init__(self, state_dim, hidden=100):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, state):
        q = F.leaky_relu(self.l1(state))
        q = F.leaky_relu(self.l2(q))
        return self.l3(q)


def quick_test_policy(env, batch_states):
    # Create the policy: pass in dims for states and actions
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    policy_dists = policy(batch_states).data
    print("policy: probs over actions for batch of states, shape: ", policy_dists.shape, ": ", policy_dists)

    # Test action selection:
    batch_actions = policy.get_action(batch_states)
    print("select a batch of actions, shape ", batch_actions.shape, ": ", batch_actions)

    # Calculate the log prob:
    log_probs = policy.log_prob(batch_states, batch_actions)
    print("log_probs, shape ", log_probs.shape, ": ", log_probs)


def quick_test_critic(env, batch_states):
    # testing the critic output
    critic = Critic(env.observation_space.shape[0])
    values = critic(batch_states)
    print("Values from critic, shape: ", values.shape, ": ", values)


# # Test:
# env = gym.make("LunarLander-v2")
#
# # create batch of states
# batch_states = torch.rand(5, env.observation_space.shape[0])
# print("batch_states, shape: ", batch_states.shape, ": ", batch_states)
# quick_test_policy(env, batch_states)
# quick_test_critic(env, batch_states)

