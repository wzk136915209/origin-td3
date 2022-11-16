import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import math
import random

random_seed = 0
from maxque import MaxQueue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.shared_a1 = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU()
        )
        self.l1 = nn.Linear(net_width, action_dim)


        self.maxaction = maxaction

    def forward(self, state):

        a = self.shared_a1(state)
        a = torch.tanh(self.l1(a)) * self.maxaction

        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        # Q1 architecture


        self.shared_q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU()
        )
        self.l1 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.shared_q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU()
        )
        self.l2 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.shared_q1(sa)
        q1 = self.l1(q1)

        q2 = self.shared_q2(sa)
        q2 = self.l2(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.shared_q1(sa)
        q1 = self.l1(q1)
        return q1


class TD3_Agent(object):
    def __init__(
            self,
            env_with_dw,
            state_dim,
            action_dim,
            max_action,
            writer,
            gamma=0.99,
            net_width=128,
            a_lr=1e-4,
            c_lr=1e-4,
            batch_size=256,
            policy_delay_freq=1,

    ):

        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(state_dim, action_dim, net_width).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.env_with_dw = env_with_dw
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005
        self.batch_size = batch_size
        self.delay_counter = -1
        self.delay_freq = policy_delay_freq

        self.human_flag = 0
        self.human_reward = 2500

        self.std = 1
        self.count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.writer = writer
        self.que = MaxQueue(15)
        self.max_norm = 15


    def prob_gass(self, mean, std, x):
        return np.array(np.exp(-np.power((x - mean), 2) / (2.0 * std ** 2)) / (std * np.sqrt(2 * np.pi)))

    def select_action(self, state):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def slect_action_normal(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a = self.actor(state)

            #sample from normal distrbution
            dist = Normal(a, self.std)
            action = dist.rsample()
            action = action.cpu().numpy()[0]
        return action


    def train(self, replay_buffer, human_replay_buffer, human_flag, episode_reward_1, human_replay_buffer_size,
              human_reward):
        self.delay_counter += 1
        self.count += 1
        with torch.no_grad():
            s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)

        agent_action = self.actor(s)
        loss = torch.nn.functional.mse_loss(agent_action, a)
        self.writer.add_scalar('loss actor', loss, self.count)
        self.writer.add_scalar('buffer bc export', replay_buffer.size, self.count)

        self.actor_optimizer.zero_grad()
        loss.backward()
        # total_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm)
        # self.writer.add_scalar('grad norm actor', total_norm_actor, self.count)
        self.actor_optimizer.step()


    def computer_q_diff(self, s, a):
        with torch.no_grad():
            s = torch.FloatTensor([s]).to(device)
            a = torch.FloatTensor([a]).to(device)
            q1, q2 = self.q_critic(s, a)
        diff = (q1 - q2).cpu().numpy()[0]
        return diff

    def save(self, EnvName, episode):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, episode))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, episode))

    def load(self, EnvName, episode):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, episode)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, episode)))


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))
        self.h_a = np.zeros((max_size, action_dim))

        self.device = device

    def add(self, state, action, reward, next_state, dead):
        # 每次只放入一个时刻的数据
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead  # 0,0,0，...，1


        self.ptr = (self.ptr + 1) % self.max_size  # 存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device),

        )
