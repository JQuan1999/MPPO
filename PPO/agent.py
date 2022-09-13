# author by 蒋权
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from model import Actor, Critic

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self, args):
        n_hidden = [32, 64, 128, 64, 32]
        self.a_update_step = args.a_update_step
        self.c_update_step = args.c_update_step
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.save_path = args.ra_ckpt_path
        self.state_dim = args.ra_state_dim + args.objective
        self.action_dim = args.ra_action_space
        self.actor = Actor(self.state_dim, self.action_dim, n_hidden).to(device)
        self.old_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(self.state_dim, args.objective, n_hidden).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.buffer = Buffer(self.batch_size)
        self.w = np.ones(3) * 1e-3
        self.critic_loss = []
        self.actor_loss = []
        self.record = None
        self.learn_step = 0
        self.train_loss_path = './log/train'

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        prob = self.actor(state).squeeze(0)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample().item()
        return action

    def cal_target(self, state_, done):
        state_ = torch.FloatTensor(state_).to(device)
        target = self.critic(state_).detach().cpu().numpy() * (1 - done)
        target_list = []
        reward = np.array(self.buffer.reward)
        for r in reward[::-1]:
            target = target + self.gamma * r
            target_list.insert(0, target.tolist())
        target_list = np.dot(np.array(target_list), self.w).tolist()
        target = torch.FloatTensor(target_list).to(device)
        return target

    def cal_advantage(self, target):
        state = torch.tensor(self.buffer.state, dtype=torch.float).to(device)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1).to(device)
        v = torch.mm(v, w).reshape(-1, )

        adv = (target - v).detach()
        return adv

    def critic_update(self, target):
        state = torch.FloatTensor(self.buffer.state).to(device)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1).to(device)
        v = torch.mm(v, w).reshape(-1, )
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(v, target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def actor_update(self, target):
        adv = self.cal_advantage(target).reshape(-1, 1)
        state = torch.FloatTensor(self.buffer.state).to(device)
        action = torch.LongTensor(self.buffer.action).view(-1, 1).to(device)

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

    def save(self, prefix=None, weight=None):
        if weight is None:
            dir_name = time.strftime('%m-%d-%H-%M')
        else:
            str_w = [str(round(100 * w)) for w in weight.reshape(-1, ).tolist()]
            dir_name = 'w' + '_'.join(str_w)

        path = '/'.join([self.save_path, dir_name])
        if not os.path.exists(path):
            os.makedirs(path)
        if prefix is None:
            actor_file = 'actor.pkl'
            critic_file = 'critic.pkl'
        else:
            actor_file = prefix + 'actor.pkl'
            critic_file = prefix + 'critic.pkl'
        actor_file = '/'.join([path, actor_file])
        critic_file = '/'.join([path, critic_file])
        torch.save(self.actor.net.state_dict(), actor_file)
        torch.save(self.critic.net.state_dict(), critic_file)

    def load(self, pkl_list):
        self.actor.net.load_state_dict(torch.load(pkl_list[0], map_location=torch.device(device)))
        self.critic.net.load_state_dict(torch.load(pkl_list[1], map_location=torch.device(device)))

    def show_loss(self):
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
        date = time.strftime('%m-%d-%H-%M')
        dir_path = '/'.join([self.train_loss_path, date])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        axes[0].plot(range(self.learn_step), self.actor_loss, color='green', label='actor_loss')
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('loss')
        axes[0].set_title('ra actor loss')
        ra_lossf = 'ra_ac_loss.log'
        file = '/'.join([self.train_loss_path, date, ra_lossf])
        np.save(file, np.array(self.actor_loss))

        axes[1].plot(range(self.learn_step), self.critic_loss, color='yellow', label='critic_loss')
        axes[1].set_title('ra critic loss')
        axes[1].set_xlabel('step')
        ra_lossf = 'ra_cr_loss.log'
        file = '/'.join([self.train_loss_path, date, ra_lossf])
        np.save(file, np.array(self.critic_loss))
        plt.show()


class Sequence_Agent:
    def __init__(self, args):
        n_hidden = [32, 64, 128, 64, 32]
        self.a_update_step = args.a_update_step
        self.c_update_step = args.c_update_step
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.save_path = args.sa_ckpt_path
        self.state_dim = args.objective + args.sa_state_dim
        self.action_dim = args.sa_action_space
        self.actor = Actor(self.state_dim, self.action_dim, n_hidden).to(device)
        self.old_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(self.state_dim, args.objective, n_hidden).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.buffer = Buffer(self.batch_size)
        self.w = None
        self.critic_loss = []
        self.actor_loss = []
        self.learn_step = 0
        self.train_loss_path = './log/train'

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        prob = self.actor(state).squeeze(0)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        return action.item()

    def cal_target(self, state_, done):
        state_ = torch.FloatTensor(state_).to(device)
        target = self.critic(state_).detach().cpu().numpy() * (1 - done)
        target[-1] = 0
        target_list = []

        reward = np.array(self.buffer.reward)

        for r in reward[::-1]:
            target = target + self.gamma * r
            target_list.insert(0, target.tolist())
        target_list = np.dot(np.array(target_list), self.w).tolist()
        target = torch.FloatTensor(target_list).to(device)
        return target

    def cal_advantage(self, target):
        state = torch.tensor(self.buffer.state, dtype=torch.float).to(device)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1).to(device)
        v = torch.mm(v, w).reshape(-1, )

        adv = (target - v).detach()
        return adv

    def actor_update(self, target):
        adv = self.cal_advantage(target).reshape(-1, 1)
        state = torch.FloatTensor(self.buffer.state).to(device)
        prob = self.actor(state)
        old_prob = self.old_actor(state)
        action = torch.LongTensor(self.buffer.action).view(-1, 1).to(device)
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
        state = torch.FloatTensor(self.buffer.state).to(device)
        v = self.critic(state)
        w = torch.from_numpy(self.w).float().unsqueeze(1).to(device)
        v = torch.mm(v, w).reshape(-1, )
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

    def save(self, prefix=None, weight=None):
        if weight is None:
            dir_name = time.strftime('%m-%d-%H-%M')
        else:
            str_w = [str(round(100 * w)) for w in weight.reshape(-1, ).tolist()]
            dir_name = 'w' + '_'.join(str_w)

        path = '/'.join([self.save_path, dir_name])
        if not os.path.exists(path):
            os.makedirs(path)
        if prefix is None:
            actor_file = 'actor.pkl'
            critic_file = 'critic.pkl'
        else:
            actor_file = prefix + 'actor.pkl'
            critic_file = prefix + 'critic.pkl'
        actor_file = '/'.join([path, actor_file])
        critic_file = '/'.join([path, critic_file])
        torch.save(self.actor.net.state_dict(), actor_file)
        torch.save(self.critic.net.state_dict(), critic_file)

    def load(self, pkl_list):
        self.actor.net.load_state_dict(torch.load(pkl_list[0], map_location=torch.device(device)))
        self.critic.net.load_state_dict(torch.load(pkl_list[1], map_location=torch.device(device)))

    def show_loss(self):
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
        date = time.strftime('%m-%d-%H-%M')
        dir_path = '/'.join([self.train_loss_path, date])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        axes[0].plot(range(self.learn_step), self.actor_loss, color='green', label='actor loss')
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('loss')
        axes[0].set_title('sa actor loss')
        ra_lossf = 'sa_ac_loss.log'
        file = '/'.join([self.train_loss_path, date, ra_lossf])
        np.save(file, np.array(self.actor_loss))

        axes[1].plot(range(self.learn_step), self.critic_loss, color='yellow', label='critic_loss')
        axes[1].set_title('sa critic loss')
        axes[1].set_xlabel('step')
        ra_lossf = 'sa_cr_loss.log'
        file = '/'.join([self.train_loss_path, date, ra_lossf])
        np.save(file, np.array(self.critic_loss))

        plt.show()