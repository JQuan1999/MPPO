# author by 蒋权
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, n_state, n_action, n_hidden):
        super(Actor, self).__init__()
        self.net = nn.ModuleList()
        for i in range(len(n_hidden)):
            if i == 0:
                self.net.append(nn.Linear(n_state, n_hidden[i]))
            else:
                self.net.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(n_hidden[-1], n_action))

    def forward(self, x):
        temp1 = x.numpy()
        for m in self.net:
            x = m(x)
        temp2 = x.detach().numpy()
        out = F.softmax(x, dim=1)
        if torch.isnan(out).any():
            print(f'origin value is {temp1}')
            print(f'value before softmax is {temp2}')
            raise Exception('out has nan value, out : {}'.format(out.detach().numpy()))

        return out


class Critic(nn.Module):
    def __init__(self, n_state, n_out, n_hidden):
        super(Critic, self).__init__()
        self.net = nn.ModuleList()
        for i in range(len(n_hidden)):
            if i == 0:
                self.net.append(nn.Linear(n_state, n_hidden[i]))
            else:
                self.net.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(n_hidden[-1], n_out))

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


