"""
Deep Q-Learning Model and Training Module
-------------------------------------------

This module implements a deep Q-learning framework using convolutional neural networks (CNNs)
for a 4x4 Tic Tac Toe reinforcement learning agent. It contains two main classes:

    1. ConvolutionalNeuralNetwork (inherits from torch.nn.Module):
       - Defines a CNN architecture with convolutional and fully-connected layers
         to approximate Q-values given a board state.
       - The forward method expects input in shape [N, 4, 4] (with N being the batch size)
         and internally adds the required channel dimension.

    2. QvalueTrainer:
       - Encapsulates the training procedure for the neural network.
       - Converts input data (states, actions, rewards, next_states, done flags) from numpy arrays to torch tensors.
       - Handles both single-sample and batch inputs by ensuring data is treated as a batch.
       - Implements the Bellman equation to update target Q-values:
         the updated reward is calculated using the next state's value according to the Bellman equation.
       - Performs backpropagation using the Adam optimizer and an MSE loss function.

Usage:
    The module is designed to be used as part of a reinforcement learning training pipeline.
    Call the QvalueTrainer.train_step() method during training to update the network parameters.
    
Note:
    This implementation focuses on the conversion of raw data to the proper tensor format and
    the integration of the Bellman update within the training step.
    
Example:
    >>> model = ConvolutionalNeuralNetwork()
    >>> trainer = QvalueTrainer(model, lr=0.001, gamma=0.9)
    >>> trainer.train_step(states, actions, rewards, next_states, done)

This module was built from scratch using PyTorch and numpy, without relying on higher-level frameworks,
to provide deeper insight into the inner workings of deep Q-learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # x shape: [1, 1, 4, 4]
            # but it waits for batch_size, channels, height, width order
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

    def forward(self, x):
        # single batch: x shape: [1, 4, 4]
        # multibatch: x shape: [N, 1, 4, 4]
        # because we put plus 1 dimension into channel index
        x = x.unsqueeze(1)  # Batch dimension + channel dimmension
        x = self.conv(x)
        x = self.fc(x)
        return x
    
class QvalueTrainer:
    """ 
    This class's train step method will train the linked neural network
    with these values states, actions, rewards, next_states, done
    """
    
    def __init__(self, model, lr, gamma, writer, player_mark):
        self.player_mark = player_mark
        self.writer = writer
        self.model_step = 0
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, done):
        # this part is for BATCH BIGGER than 1 row
        # # len(N,4,4) == 3 (batches)
        # the incoming data is converted into np.array to make it faster for pytorch
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        # for more than one batch done will be just a list, so later
        # we can iterate through it (len(done))

        # this part is for SINGLE BATCH
        # If the input is a single sample (not a batch), we convert it to the proper format
        # Even a single example must be treated as a batch
        # In this case, we don't need to manually sum or stack numpy arrays into a torch tensor
        if len(states.shape) == 2: # len(4,4) == 2 (True)
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            done = (done, )

        # prediction of the model
        pred = self.model(states)

        with torch.no_grad():
            target_predictions = pred.clone()
            for idx in range(len(done)):
                # state result corresponding reward
                new_reward = rewards[idx]
                # if the game continued (not terminated)
                # it is important to have next state for the Bellmann equation
                if not done[idx]:
                    next_state_single = next_states[idx].unsqueeze(0) # this solely batch has to have the format (1, 4, 4)
                    # grabbing the best predicted value from the corresponding next state
                    new_reward = rewards[idx] + self.gamma * torch.max(self.model(next_state_single))
                #  the updated reward is calculated using the next state's value according to the Bellman equation
                target_predictions[idx][actions[idx]] = new_reward
    
        # simple pytorch backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target_predictions, pred)
        # derivation is used for loss (whatever it is inside that (usually used for criterion (MSE)))
        loss.backward()

        # recalculate the weights
        self.optimizer.step()
        self.model_step += 1
        
        if self.player_mark == 1:
            self.writer.add_scalar("Loss_X", loss.item(), self.model_step)
        elif self.player_mark == -1:
            self.writer.add_scalar("Loss_O", loss.item(), self.model_step)