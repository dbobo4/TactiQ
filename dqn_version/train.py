"""
TactiQ Training Script
-----------------------

This script implements the training loop for a self-play reinforcement learning project 
for a 4x4 Tic Tac Toe game using Deep Q-Learning with convolutional neural networks (CNNs).

Key Features:
- Two agents (agent_x and agent_o) are created with shared environment.
- Each agent is trained using short-term and long-term memory updates.
- Custom memory is managed using the LastMemory helper class.
- The script runs for a specified number of episodes and saves the trained models at the end.

Usage:
    Run this script from the project root directory:
        python train.py
"""

import torch
from agent import Agent
from TicTacToeEnv import TicTacToeEnv
from torch.utils.tensorboard.writer import SummaryWriter
import copy
from lastmemory import LastMemory

ILLEGAL_MOVE_LIMIT = 100
MAX_EPISODES = 10_000

# it can be used for logging later
# writer = SummaryWriter("runs/common_run")

env = TicTacToeEnv()

# creation of the two agent
# with shared environment (architecture)
agent_x = Agent(env, +1)  # X is +1
agent_o = Agent(env, -1)  # O is -1

# lastMemory object Initialization for both of the agent
agent_x_memory = LastMemory()
agent_o_memory = LastMemory()

episode_count = 0
while episode_count < MAX_EPISODES:
    done_x = None
    done_o = None
    
    x_illegal_move_cnt = 0
    o_illegal_move_cnt = 0
    
    is_x_has_reached_max_illegal_move = False
    is_o_has_reached_max_illegal_move = False

    while True:
        state_old = agent_x.get_state()
        move = agent_x.get_action(state_old)
        reward_x, done_x, is_illegal_move_x = env.play_step(move, agent_x.player_mark)
        state_new = agent_x.get_state()
        agent_x.train_short_memory(state_old, move, reward_x, state_new, done_x)
        agent_x.remember(state_old, move, reward_x, state_new, done_x)
        
        # we need last memory to make remember the agent which lost the state which caused the other player to win
        # that's why it is called last memory (and because an array is reference type we need to use this method)
        agent_x_memory.state_old = copy.deepcopy(state_old)
        agent_x_memory.action = move
        agent_x_memory.reward = reward_x
        agent_x_memory.state_new = copy.deepcopy(state_new)
        agent_x_memory.done = done_x
        
        if is_illegal_move_x:
            x_illegal_move_cnt += 1
        if not is_illegal_move_x or (x_illegal_move_cnt >= ILLEGAL_MOVE_LIMIT):
            if (x_illegal_move_cnt >= ILLEGAL_MOVE_LIMIT):
                is_x_has_reached_max_illegal_move = True
            break
            
    if not done_x and (not is_x_has_reached_max_illegal_move):    
        while True:
            state_old = agent_o.get_state()
            move = agent_o.get_action(state_old)
            reward_o, done_o, is_illegal_move_o = env.play_step(move, agent_o.player_mark)
            state_new = agent_o.get_state()
            agent_o.train_short_memory(state_old, move, reward_o, state_new, done_o)
            agent_o.remember(state_old, move, reward_o, state_new, done_o)
            
            agent_o_memory.state_old = copy.deepcopy(state_old)
            agent_o_memory.action = move
            agent_o_memory.reward = reward_o
            agent_o_memory.state_new = copy.deepcopy(state_new)
            agent_o_memory.done = done_o
            
            if is_illegal_move_o:
                o_illegal_move_cnt += 1
            if not is_illegal_move_o or (o_illegal_move_cnt >= ILLEGAL_MOVE_LIMIT):
                if (o_illegal_move_cnt >= ILLEGAL_MOVE_LIMIT):
                    is_o_has_reached_max_illegal_move = True
                break
    
    # Need to use penalty for the correspinding player if the other player has won
    # but the memory of the game should be the last played memory
    if done_x:
        agent_o.train_short_memory(agent_o_memory.state_old, 
                                   agent_o_memory.action, 
                                   -10, 
                                   agent_o_memory.state_new, 
                                   False)
    if done_o:
        agent_x.train_short_memory(agent_x_memory.state_old, 
                                    agent_x_memory.action, 
                                    -10, 
                                    agent_x_memory.state_new, 
                                    False)

    if done_x or done_o or is_x_has_reached_max_illegal_move or is_o_has_reached_max_illegal_move:
        agent_x.step += 1
        agent_x.train_long_memory()
        
        agent_o.step += 1
        agent_o.train_long_memory()
        
        # new turn -> reset last memory
        agent_x_memory.reset()
        agent_o_memory.reset()
        
        env.reset()
    
    print(f"Episode {episode_count}, X_reward={reward_x:.3f}, O_reward={reward_o:.3f}")

    episode_count += 1
    print('episode counter has been increased by 1')

# save models after finished with the training
torch.save(agent_x.model.state_dict(), 'trained_model_X.pth')
torch.save(agent_o.model.state_dict(), 'trained_model_O.pth')
print("Training completed and both models have been saved successfully!")