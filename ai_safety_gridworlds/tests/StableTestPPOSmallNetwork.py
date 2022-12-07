from typing import Callable

import ai_safety_gridworlds
import ai_safety_gridworlds.helpers.factory
from ai_safety_gridworlds.tests.GridWrapper import BasicWrapper
from stable_baselines3 import A2C, DDPG, DQN, HER, SAC, TD3
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.results_plotter import ts2xy
import os


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    i = 0
    for log in log_folder:
        x, y = ts2xy(load_results(log), 'timesteps')
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(title)
        plt.plot(x, y,label = 'seed:' )
        i += 1
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.legend()
    plt.show()


environment_name = 'distributional_shift'

env = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name,is_testing=True)
env.reset()
wrappedEnv = BasicWrapper(env)
seeds = [11]
seedsStr = []
LogPath = 'F:/ai-safety-gridworlds/log/SmallNetworkTest/'
LogDirs = []
for seed in seeds:
    seedsStr.append(str(seed))
    path = LogPath + str(seed)

    LogDirs.append(path)

    #os.mkdir(path)
    #monitor = Monitor(wrappedEnv, path)
    #model = PPO('MlpPolicy', monitor, verbose=1,seed=seed,learning_rate=linear_schedule(0.005))
    #model.learn(total_timesteps=100000,eval_log_path=path)
    #model.save(path=LogPath+"PPO0602Multi")



plot = results_plotter.plot_results(labels=seedsStr,dirs=LogDirs,num_timesteps=100000, x_axis=results_plotter.X_TIMESTEPS, task_name="PPO")
#plot_results(['F:/ai-safety-gridworlds/log/log1','F:/ai-safety-gridworlds/log/log2'])

TestEnv = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name,is_testing=True)
TestEnv.reset()

wrappedTestEnv = BasicWrapper(TestEnv)
model = PPO.load(LogPath+"PPO0602Multi", env=wrappedTestEnv)
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
#print(mean_reward, ' ', std_reward)

obs = wrappedTestEnv.reset()
i = 0
OReward = 0
Rewards = []
while i < 100:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = wrappedTestEnv.step(action)
    OReward += rewards
    if done:
        Rewards.append(OReward)
        print(OReward)
        OReward = 0
        print(obs)

        obs = wrappedTestEnv.reset()
        i += 1

i = 1
x= []
for value in Rewards:
    x.append(i)
    i +=1
    y = value
plt.scatter(x, Rewards,label = 'seed:' )

plt.xlabel('Number of Timesteps')
plt.ylabel('Rewards')
plt.show()