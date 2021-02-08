from gym import make as gym_make
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import wandb


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    complete_episodes = 0
    episode_10_list = np.zeros(10)
    episode_final = False 
    try:
        with tqdm(range(max_episodes),leave=False) as pbar:
            for episode, ch in enumerate(pbar):
                pbar.set_description("[Train] Episode %d" % episode)
                state = env.reset()
                episode_reward = 0    

                for step in range(max_steps):
                    action = agent.get_action(state, episode)
                    #---------------------------------------------------------------------
                    next_state, _, done, _ = env.step(action)
                    if done:
                        # next_state = None
                        # 直近10episodeの立てたstep数リストに追加
                        episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                        if step < 195:
                            reward = -1
                            complete_episodes = 0
                        else:
                            reward = 1
                            complete_episodes += 1
                    else:
                        reward = 0

                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    td_error = agent.get_td_error(state, action, next_state, reward)
                    agent.td_error_memory.push(0)
                    episode_reward += reward
                    state = next_state

                    # update the agent if enough transitions are stored in replay buffer
                    if len(agent.replay_buffer) > batch_size:
                        agent.update(batch_size,episode)

                    if done:
                        episode_rewards.append(episode_reward)
                        wandb.log({ "episode reward": episode_reward,
                                    "max steps":step+1,
                                    "mean steps of past 10 trials": episode_10_list.mean()})
                        agent.update_td_error_memory()
                        print('%d Episode: Finished after %d steps：10試行の平均step数 = %.1lf' % (episode, step + 1, episode_10_list.mean()))
                        break

                if episode_final is True:
                    # 動画描画をコメントアウトしています
                    # 動画を保存と描画
                    #display_frames_as_gif(frames)
                    break

                # 10連続で200step経ち続けたら成功
                if complete_episodes >= 10:
                    pass
                    # print('10回連続成功')
                    # episode_final = True  # 次の試行を描画を行う最終試行とする

    except KeyboardInterrupt:
        print('Training stopped manually!!!')
        pass

    return episode_rewards