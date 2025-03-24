"""
Agent Module for Self-Play Reinforcement Learning
---------------------------------------------------

This module defines the Agent class, which is used in a self-play reinforcement learning 
framework. The Agent interacts with a provided environment, maintains a replay memory, and uses a 
deep Q-learning approach with a convolutional neural network (CNN) 
to learn an effective policy for playing the game.

Key Features:
    - Replay Memory: Uses a deque for efficient memory management (up to MAX_MEMORY samples).
    - Deep Q-Learning: Employs a CNN-based model (from the model module) to approximate Q-values.
    - Training: Provides methods for both short-term (single transition) and long-term (mini-batch) training updates 
      using the QvalueTrainer.
    - Action Masking: Implements epsilon-greedy exploration with action masking to ensure only valid moves 
      (i.e., empty board cells) are considered.
    - Epsilon Decay: Adjusts the exploration rate gradually based on the number of training steps.

Usage Example:
    >>> from agent import Agent
    >>> agent = Agent(env, player_mark=+1)
    >>> current_state = agent.get_state()
    >>> action = agent.get_action(current_state)
    >>> agent.remember(state, action, reward, next_state, done)
    >>> agent.train_short_memory(state, action, reward, next_state, done)
    >>> agent.train_long_memory()

This module is implemented using PyTorch and numpy and is designed to be integrated into a 
self-play reinforcement learning pipeline for training agents.
"""

import torch
import random
from collections import deque
from model import ConvolutionalNeuralNetwork, QvalueTrainer
from torch.utils.tensorboard import SummaryWriter # for logging in Tensorboard

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

writer = SummaryWriter(log_dir="dqn_version/runs/common_run")

class Agent:
    def __init__(self, env, player_mark):
        self.step = 0
        self.player_mark = player_mark
        self.env = env
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # deque() to handle memory efficiency
        # self.writer = writer # common tensorboard logging
        self.model = ConvolutionalNeuralNetwork()
        self.trainer = QvalueTrainer(self.model, LEARNING_RATE, self.gamma, writer)
    
    def get_state(self):
        actual_board_state = self.env.get_board_actual_state()
        
        # if we would like, we could give back not the board, but other parameters
        # keywords: how far, which side etc...
        # state = []
        # return np.array(state, dtype=int)

        return actual_board_state
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
        
    # Here we are going to use ACTION MASKING to deal with occupied places effectively
    def get_action(self, state, exploit_only=False):
        # Update epsilon to gradually shift from exploration to exploitation
        self.epsilon = max(0, 1000 - self.step)

        # valid = list of legal action indices (e.g. [0, 2, 4])
        valid = [i for i, v in enumerate(state.flatten()) if v == 0]

        # Exploration: pick a random valid move with probability proportional to epsilon
        if not exploit_only and random.randint(0, 400) < self.epsilon:
            return random.choice(valid)

        # Exploitation: choose the valid action with the highest Q‑value
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        # self.model(state_t) → shape: (batch_size, num_actions)
        # batch_size=1 → (1, 16); [0] selects the first (and only) row → shape (16,)
        q_vals = self.model(state_t)[0]

        # Create a mask filled with -inf so invalid moves cannot be selected
        mask = torch.full_like(q_vals, float('-inf'))
        mask[valid] = q_vals[valid]

        return int(torch.argmax(mask).item())