
from collections import namedtuple
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim

import ai_safety_gridworlds
from ai_safety_gridworlds.helpers import factory

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 99





class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def get_state(board):
    for row in range(board.shape[0]):
        for column in range(board.shape[1]):
            if board[row][column] == 2: return np.array([row, column])

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []

    time_step = env.reset()
    obs = get_state(time_step.observation['board'])
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_time_step = env.step(action)
        next_obs = get_state(next_time_step.observation['board'])
        reward = next_time_step.reward
        is_done = next_time_step.step_type == 2

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []

            time_step = env.reset()
            next_obs = get_state(time_step.observation['board'])
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = ai_safety_gridworlds.helpers.factory.get_environment_obj('distributional_shift')
    # get size of state and action from environment
    state_size = 2
    action_size = env.action_spec().maximum + 1
    state = env.reset()
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)


    net = Net(state_size, HIDDEN_SIZE, action_size)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
