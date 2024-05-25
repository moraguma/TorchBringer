import gymnasium as gym
import math
import random
from itertools import count
from aim import Run

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
from components.replay_memory import ReplayMemory, Transition
import components.builders as builders
import modules.module_utils as mu
from components.epsilon import Epsilon

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN():
    """
    Deep Q Network - built based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    config spec
    {
        "action_space": action space spec (read builders.py) -> The environment's action space
        "gamma": float -> Value of gamma
        "tau": float -> Value of tau
        "epsilon": epsilon spec (read builders.py) -> Epsilon
        "batch_size": int -> Batch size
        "grad_clip_value": float -> Value to clip gradient. No clipping if not specified
        "loss": loss spec (read builders.py) -> Loss
        "optimizer": optimizer spec (read builders.py) -> Optimizer
        "replay_buffer_size": int -> capacity of the replay buffer
        "network": List of layer specs (read builders.py) -> Sequential network that will be used
    }
    """

    def __init__(self, config):
        self.action_space = builders.build_action_space(config["action_space"])

        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.epsilon: Epsilon = builders.build_epsilon(config["epsilon"])
        self.batch_size = config["batch_size"]
        self.grad_clip_value = mu.value_or_none(config, "grad_clip_value")

        self.policy_net = builders.build_sequential(config["network"]).to(device)
        self.target_net = builders.build_sequential(config["network"]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(config["replay_buffer_size"])
        self.loss = builders.build_loss(config["loss"])
        self.optimizer = builders.build_optimizer(self.policy_net, config["optimizer"])

        self.past_state = None
        self.past_action = None

        self.steps_done = 0

    def step(self, state, reward, terminal):
        self.experience(state, reward, terminal)
        self.optimize()
        self.past_action = self.select_action(state)
        
        return self.past_action


    def experience(self, state, reward, terminal):
        if not self.past_state is None:
            self.memory.push(self.past_state, self.past_action, state, reward)
        
        if terminal:
            self.past_state = None
        else:
            self.past_state = state

        
    def optimize(self):
        if len(self.memory) >= self.batch_size:
            # Transpose batch
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            # Mask used to consider only non-terminal states
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) for state action pairs in batch
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute the expected Q values (r +  γ max_a ​Q(s′,a))
            next_state_values = torch.zeros(self.batch_size, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute loss
            step_loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            step_loss.backward()

            if not self.grad_clip_value is None:
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip_value)
            self.optimizer.step()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon()
        if sample > eps_threshold: # Action with largest Q value
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else: # Random action
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)