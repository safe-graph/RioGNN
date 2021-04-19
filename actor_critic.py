import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
    Actor-Critic implementations
    Paper: Actor-Critic Algorithms
    Source: 
"""

# torch.backends.cudnn.enabled = False  # Non-deterministic algorithm

class PGNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Actor(object):
    # dqn Agent
    def __init__(self, state_dim, action_dim, device, LR):
        # Dimensions of state space and action space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LR = LR

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(self.device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            #prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
            prob_weights = F.softmax(network_output, dim=0).data.cpu().numpy()
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    def learn(self, state, action, td_error):
        self.time_step += 1
        # Step 1: Forward propagation
        softmax_input = self.network.forward(torch.FloatTensor(state).to(self.device)).unsqueeze(0)
        action = torch.LongTensor([action]).to(self.device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')

        # Step 2: Backpropagation
        # Here you need to maximize the value of the current strategy, so you need to maximize "neg_log_prob * tf_error", that is, minimize "-neg_log_prob * td_error"
        loss_a = -neg_log_prob * td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Critic(object):

    def __init__(self, state_dim, action_dim, device, LR, GAMMA):
        # Dimensions of state space and action space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LR = LR
        self.GAMMA = GAMMA

        # init network parameters
        self.network = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()


    def train_Q_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state).to(self.device), torch.FloatTensor(next_state).to(self.device)
        # Forward propagation
        v = self.network.forward(s)     # v(s)
        v_ = self.network.forward(s_)   # v(s')

        # Backpropagation
        loss_q = self.loss_func(reward + self.GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward + self.GAMMA * v_ - v

        return td_error






