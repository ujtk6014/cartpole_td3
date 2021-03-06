import os

import math
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import datetime
import gym
import wandb

from network import DDQNAgent
from utils import *


def train():
    
    #logger
    wandb.init(project='cartpole-continuous',
        config={
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_episodes": 500,
        "max_steps": 200,
        "gamma": 0.99,
        "buffer_maxlen": 10000,
        "prioritized_on": True,
        "State": 'angle,ang_vel',}
    )
    config = wandb.config

    max_episodes = config.max_episodes
    max_steps = config.max_steps
    batch_size = config.batch_size

    gamma = config.gamma
    buffer_maxlen = config.buffer_maxlen
    learning_rate = config.learning_rate
    prioritized_on = config.prioritized_on

    # simulation of the agent solving the spacecraft attitude control problem
    env = gym.make("CartPole-v0")
    agent = DDQNAgent(env, gamma, buffer_maxlen, learning_rate, True, max_episodes * max_steps, prioritized_on)
    wandb.watch([agent.q_net,agent.q_net_target], log="all")
    #学習済みモデルを使うとき
    #curr_dir = os.path.abspath(os.getcwd())
    #agent = torch.load(curr_dir + "/models/spacecraft_control_ddqn_hist.pkl")
    episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size, prioritized_on)

    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 10 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 10 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 10 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    #--------------------------------------------------------------  
    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    # plt.show()

    date = datetime.datetime.now()
    date = '{0:%Y%m%d}'.format(date)
    curr_dir = os.path.abspath(os.getcwd())
    # plt.savefig(curr_dir + "/plot_reward_"+ date + ".png")
    if not os.path.isdir("models"):
        os.mkdir("models")
    # torch.save(agent, curr_dir + "/models/ddqn_agent.pkl")


if __name__ == '__main__':
    plt.close()
    train()
