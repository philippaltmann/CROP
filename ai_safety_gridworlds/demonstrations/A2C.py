import platform
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

# hyperparameters
hidden_size = 50
learning_rate = 3e-4

# Constants
GAMMA = 0.8
num_steps = 300
max_episodes = 10000

class ActorCritic(nn.Module):
    def __init__(self, env, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

        self.rows, self.columns = env.observation_spec()['board'].shape

    def forward(self, board):
        state = self.get_state(board)
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

    def get_state(self, board):
        formed_board = np.zeros(63)
        i = 0
        for row in range(self.rows):
            for column in range(self.columns):
                #if board[row][column] == 2: return np.array([row, column])
                formed_board[i] = board[row][column]
                i += 1
        return formed_board


def a2c(env):
    #num_inputs = env.observation_space.shape[0]
    num_inputs = 63 #(x,y) for gridworld
    num_outputs = 4

    actor_critic = ActorCritic(env, num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []
        actions= []

        board = env.reset().observation['board']
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(board)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            time_step = env.step(action)
            new_board = time_step.observation['board']
            reward = time_step.reward
            done = time_step.step_type == 2 # 2 == terminated in gridworld, 1 == midgame

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            board = new_board
            actions.append(action)

            if done or steps == num_steps - 1:
                Qval, _ = actor_critic.forward(new_board)
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                #print("steps needed: {}\n".format(steps))
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {}\n".format(episode,
                                                                                                               np.sum(
                                                                                                                   rewards),
                                                                                                               steps,
                                                                                                               average_lengths[
                                                                                                                 -1]))

                    #print("[{}]".format(', '.join(map('{:.2f}'.format, actions))))
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.01 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()