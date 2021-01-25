import os

import math
import matplotlib.pyplot as plt
import torch
from gym import wrappers 
import numpy as np


from network import TD3Agent
from utils import *


def train(batch_size=128, critic_lr=1e-3, actor_lr=1e-4, max_episodes=1000, max_steps= 500, gamma=0.99, tau=1e-3,
          buffer_maxlen=100000):
    # simulation of the agent solving the spacecraft attitude control problem
    env = make("CartPoleSwingUpContinuous")

    max_episodes = max_episodes
    max_steps = max_steps
    batch_size = batch_size

    gamma = gamma
    tau = tau
    buffer_maxlen = buffer_maxlen
    critic_lr = critic_lr
    actor_lr = actor_lr

    agent = TD3Agent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr, True, max_episodes * max_steps)
    episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)

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
    tate = 4.0
    yoko = 6.0
    #--------------------------------------------------------------  
    curr_dir = os.path.abspath(os.getcwd())
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    # plt.show()
    plt.savefig(curr_dir + "/results/reward_hist.png")

    if not os.path.isdir("models"):
        os.mkdir("models")
    torch.save(agent, curr_dir + "/models/cartpole_swingup_ddpg.pkl")


def evaluate():
    # simulation of the agent solving the cartpole swing-up problem
    env = make("CartPoleSwingUpContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)


    curr_dir = os.path.abspath(os.getcwd())
    agent = torch.load(curr_dir + "/models/cartpole_swingup_ddpg.pkl")#, map_location = torch.device('cpu'))
    agent.train = False

    state = env.reset()
    r = 0
    theta = []
    actions = []

    for i in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        actions.append(action)
        # env.render()
        theta.append(math.degrees(next_state[2]))
        r += reward
        state = next_state

    env.close()

    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")

    plt.figure()
    plt.plot(theta)
    plt.title('Angle')
    plt.ylabel('Angle in degrees')
    plt.xlabel('Time step t')
    plt.savefig(curr_dir + "/results/plot_angle.png")

    plt.figure()
    plt.plot(actions)
    plt.title('Action')
    plt.ylabel('Action in Newton')
    plt.xlabel('Time step t')
    plt.savefig(curr_dir + "/results/plot_action.png")
    plt.show()
  

if __name__ == '__main__':
    plt.close()
    val = input('Enter the number 1:train 2:evaluate  > ')
    if val == '1':
        train()
    elif val == '2':
        evaluate()
    else:
        print("You entered the wrong number, run again and choose from 1 or 2 or 3.")
