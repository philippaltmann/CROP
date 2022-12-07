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

env = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name)
env.reset()
wrappedEnv = BasicWrapper(env)
seeds = [11]
seedsStr = []
LogPath = 'F:/ai-safety-gridworlds/log/A2CStandardTest/'
LogDirs = []
for seed in seeds:
    seedsStr.append(str(seed))
    path = LogPath + str(seed)

    LogDirs.append(path)

    os.mkdir(path)
    monitor = Monitor(wrappedEnv, path)
    model = A2C('MlpPolicy', monitor, verbose=1,seed=seed,rms_prop_eps=0.1,ent_coef=0.1,n_steps=3,learning_rate=linear_schedule(0.005))
    model.learn(total_timesteps=500000,eval_log_path=path)
    model.save("a2c_Lava")


plot = results_plotter.plot_results(labels=seedsStr,dirs=LogDirs,num_timesteps=500000, x_axis=results_plotter.X_TIMESTEPS, task_name="A2C")
#plot_results(['F:/ai-safety-gridworlds/log/log1','F:/ai-safety-gridworlds/log/log2'])

TestEnv = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name,is_testing=True)
TestEnv.reset()

wrappedTestEnv = BasicWrapper(TestEnv)
model = DQN.load("a2c_Lava", env=wrappedTestEnv)

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
        OReward = 0
        obs = wrappedTestEnv.reset()
        i += 1

i = 1
x= []
for value in Rewards:
    x.append(i)
    i +=1
    y = value
plt.scatter(x, Rewards,label = 'seed:' )

plt.xlabel('Number of Episodes')
plt.ylabel('Rewards')
plt.show()