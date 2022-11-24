import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.sh1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc1 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        out = self.sh1(cat)
        return torch.sigmoid(self.fc1(out))

class GAIL:
    def __init__(self, agent, state_dim, action_dim, writer, hidden_dim=256, lr_d=1e-4):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.writer = writer
        self.count = 0
        self.batch_size = 256

    # def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
    def learn(self, buffer_human, buffer_agent, agent_s, agent_a, next_s, dones):

        self.count += 1
        expert_states, expert_actions = buffer_human.sample_export(self.batch_size)

        agent_states = torch.FloatTensor([agent_s]).to(device)
        agent_actions = torch.FloatTensor([agent_a]).to(device)


        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        expert_prob = expert_prob.clip(1e-6, 1.0)
        agent_prob = agent_prob.clip(1e-6, 1.0)

        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob))\
                           + nn.BCELoss()(
            expert_prob, torch.zeros_like(expert_prob))

        self.writer.add_scalar('loss discriminator', discriminator_loss, self.count)
        self.writer.add_scalar('prob agent', agent_prob.mean(), self.count)
        self.writer.add_scalar('prob export', expert_prob.mean(), self.count)


        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).squeeze().detach().cpu().numpy()

        # state, action, reward, next_state, dead
        buffer_agent.add(agent_s, agent_a, rewards, next_s, dones)
        self.agent.train(buffer_agent)
    def reward(self, s, a):
        agent_states = torch.FloatTensor(s).to(device)
        agent_actions = torch.FloatTensor(a).to(device)
        agent_prob = self.discriminator(agent_states, agent_actions)
        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        return rewards