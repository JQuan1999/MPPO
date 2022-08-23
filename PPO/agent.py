# author by 蒋权
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from model.model import Actor,Critic


class Buffer:
    def __init__(self, batch_size):
        self.state = []
        self.action = []
        self.reward = []
        self.state_ = []
        self.done = []
        self.cnt = 0
        self.batch_size = batch_size

    def store(self, state, action, reward, state_, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.state_.append(state_)
        self.done.append(done)
        self.cnt += 1

    def clean(self):
        self.cnt = 0
        self.state = []
        self.action = []
        self.reward = []
        self.state_ = []
        self.done = []


class Route_Agent:
    def __init__(self, n_state, n_action, args):
        n_hidden = [32, 64, 64, 32]
        self.a_update_step = args.a_update_step
        self.c_update_step = args.c_update_step
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.state_dim = n_state
        self.action_dim = n_action
        self.actor = Actor(n_state, n_action, n_hidden)
        self.old_actor = copy.deepcopy(self.actor)
        self.critic = Critic(n_state, args.objective, n_hidden)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.buffer = Buffer(self.batch_size)
        self.w = None
        self.critic_loss = []
        self.actor_loss = []
        self.record = None
        self.learn_step = 0

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        prob = self.actor(state).squeeze(0)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample().item()
        return action

    def cal_target(self, state_, done):
        state_ = torch.FloatTensor(state_)
        target = self.critic(state_).detach().numpy() * (1 - done)
        target = np.dot(target, self.w)
        target_list = []
        reward = np.array(self.buffer.reward)
        min_r = np.tile(np.min(reward, axis=0), (reward.shape[0], 1))
        max_r = np.tile(np.max(reward, axis=0), (reward.shape[0], 1))
        norm_reward = (reward - min_r) / (max_r - min_r + 1e-6)
        for r in norm_reward[::-1]:
            r = np.dot(r, self.w)
            target = target + self.gamma * r
            target_list.insert(0, target)
        target = torch.FloatTensor(target_list)
        return target

    def cal_advantage(self, target):
        state = torch.tensor(self.buffer.state, dtype=torch.float)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1)
        v = torch.mm(v, w).squeeze()

        adv = (target - v).detach()
        return adv

    def critic_update(self, target):
        state = torch.FloatTensor(self.buffer.state)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1)
        v = torch.mm(v, w).squeeze()
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(v, target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def actor_update(self, target):
        adv = self.cal_advantage(target).reshape(-1, 1)
        state = torch.FloatTensor(self.buffer.state)
        action = torch.LongTensor(self.buffer.action).view(-1, 1)

        prob = self.actor(state).gather(1, action)
        old_prob = self.old_actor(state).gather(1, action)

        ratio = torch.exp(torch.log(prob) - torch.log(old_prob))
        surr = ratio * adv
        loss = - torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon,
                                                        1 + self.epsilon) * adv))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return loss.item()

    def learn(self, state_, done):
        target = self.cal_target(state_, done)
        self.old_actor.load_state_dict(self.actor.state_dict())
        actor_loss = 0
        critic_loss = 0
        for _ in range(self.a_update_step):
            actor_loss += self.actor_update(target)
        for _ in range(self.c_update_step):
            critic_loss += self.critic_update(target)
        self.buffer.clean()
        self.learn_step += 1
        actor_loss = actor_loss / self.a_update_step
        critic_loss = critic_loss / self.c_update_step
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)
        return actor_loss, critic_loss

    def store(self, state, action, reward, done):
        if self.record is None:
            self.record = [state, action, reward, done]
        else:
            s = self.record[0]
            a = self.record[1]
            r = self.record[2]
            d = self.record[3]
            self.buffer.store(s, a, r, state, d)
            if done is True:
                state_ = np.zeros(self.state_dim).tolist()
                self.buffer.store(state, action, reward, state_, done)
            self.record = [state, action, reward, done]

    def cweight(self, weight):
        self.w = weight

    def save(self, file):
        path = './param/ra/'
        date = time.strftime('%Y-%m-%d-%H-%M-%S')
        path = path + date
        if not os.path.exists(path):
            os.makedirs(path)
        actor_file = file + 'actor.pkl'
        critic_file = file + 'critic.pkl'
        actor_file = '/'.join([path, actor_file])
        critic_file = '/'.join([path, critic_file])
        torch.save(self.actor.net.state_dict(), actor_file)
        torch.save(self.actor.net.state_dict(), critic_file)

    def load(self, pkl_list):
        state_dict = torch.load(pkl_list[0])
        print(state_dict)
        self.actor.net.load_state_dict(torch.load(pkl_list[0]))
        state_dict = torch.load(pkl_list[1])
        print(state_dict)
        self.critic.net.load_state_dict(torch.load(pkl_list[1]))

    def show_loss(self):
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
        axes[0].plot(range(self.learn_step), self.actor_loss, color='green', label='actor_loss')
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('loss')
        axes[0].set_title('sequence agent actor loss')
        axes[1].plot(range(self.learn_step), self.critic_loss, color='yellow', label='critic_loss')
        axes[1].set_title('sequence agent critic loss')
        axes[1].set_xlabel('step')
        axes[1].set_ylabel('loss')
        plt.show()


class Sequence_Agent:
    def __init__(self, n_state, n_action, args):
        n_hidden = [32, 64, 64, 32]
        self.a_update_step = args.a_update_step
        self.c_update_step = args.c_update_step
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.actor = Actor(n_state, n_action, n_hidden)
        self.old_actor = copy.deepcopy(self.actor)
        self.critic = Critic(n_state, args.objective, n_hidden)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.buffer = Buffer(self.batch_size)
        self.w = None
        self.critic_loss = []
        self.actor_loss = []
        self.learn_step = 0

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        prob = self.actor(state).squeeze(0)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        return action.item()

    def cal_target(self, state_, done):
        state_ = torch.FloatTensor(state_)
        target = self.critic(state_).detach().numpy() * (1 - done)
        target = np.dot(target, self.w)
        target_list = []

        reward = np.array(self.buffer.reward)
        min_r = np.tile(np.min(reward, axis=0), (reward.shape[0], 1))
        max_r = np.tile(np.max(reward, axis=0), (reward.shape[0], 1))
        norm_reward = (reward - min_r) / (max_r - min_r + 1e-6)

        for r in norm_reward[::-1]:
            r = np.dot(r, self.w)
            target = target + self.gamma * r
            target_list.insert(0, target)
        target = torch.FloatTensor(target_list)
        return target

    def cal_advantage(self, target):
        state = torch.tensor(self.buffer.state, dtype=torch.float)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1)
        v = torch.mm(v, w).squeeze()

        adv = (target - v).detach()
        return adv

    def actor_update(self, target):
        adv = self.cal_advantage(target).reshape(-1, 1)
        state = torch.FloatTensor(self.buffer.state)
        prob = self.actor(state)
        old_prob = self.old_actor(state)
        action = torch.LongTensor(self.buffer.action).view(-1, 1)
        prob = prob.gather(1, action)
        old_prob = old_prob.gather(1, action)

        ratio = torch.exp(torch.log(prob) - torch.log(old_prob))
        surr = ratio * adv
        loss = - torch.mean(torch.min(surr,
                                      torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * adv))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return loss.item()

    def critic_update(self, target):
        state = torch.FloatTensor(self.buffer.state)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1)
        v = torch.mm(v, w).squeeze()
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(v, target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def learn(self, state_, done):
        target = self.cal_target(state_, done)
        self.old_actor.load_state_dict(self.actor.state_dict())
        actor_loss = 0
        critic_loss = 0
        for _ in range(self.a_update_step):
            actor_loss += self.actor_update(target)
        for _ in range(self.c_update_step):
            critic_loss += self.critic_update(target)
        self.learn_step += 1
        self.buffer.clean()
        actor_loss = actor_loss / self.a_update_step
        critic_loss = critic_loss / self.c_update_step
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)
        return actor_loss, critic_loss

    def cweight(self, weight):
        self.w = weight

    def save(self, file):
        path = './param/sa/'
        date = time.strftime('%Y-%m-%d-%H-%M-%S')
        path = path + date
        if not os.path.exists(path):
            os.makedirs(path)
        actor_file = file + 'actor.pkl'
        critic_file = file + 'critic.pkl'
        actor_file = '/'.join([path, actor_file])
        critic_file = '/'.join([path, critic_file])
        torch.save(self.actor.net.state_dict(), actor_file)
        torch.save(self.critic.net.state_dict(), critic_file)

    def load(self, pkl_list):
        self.actor.net.load_state_dict(torch.load(pkl_list[0]))
        self.critic.net.load_state_dict(torch.load(pkl_list[1]))

    def show_loss(self):
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
        axes[0].plot(range(self.learn_step), self.actor_loss, color='green', label='actor loss')
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('loss')
        axes[0].set_title('sequence agent actor loss')
        axes[1].plot(range(self.learn_step), self.critic_loss, color='yellow', label='critic_loss')
        axes[1].set_title('sequence agent critic loss')
        axes[1].set_xlabel('step')
        axes[1].set_ylabel('loss')
        plt.show()