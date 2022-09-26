# author by 蒋权
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from agent import Sequence_Agent, Route_Agent
from env import PPO_ENV
from utils.uniform_weight import init_weight, cweight
from utils.config import config


def show_reward(sa_r, ra_r):
    path = './log/reward'
    date = time.strftime('%m-%d-%H-%M')
    dir_path = '/'.join([path, date])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    _, axes = plt.subplots(1, 2)
    axes[0].plot(range(len(sa_r)), sa_r)
    sa_rf = '/'.join([path, date, 'sa_rlog'])
    np.save(sa_rf, np.array(sa_r))

    axes[1].plot(range(len(ra_r)), ra_r)
    ra_rf = '/'.join([path, date, 'ra_rlog'])
    np.save(ra_rf, np.array(ra_r))
    plt.show()


def show_objective(objs):
    _, axes = plt.subplots(1, 3)
    axes[0].plot(range(objs.shape[0]), objs[:, 0])
    axes[0].set_title('use ratio')

    axes[1].plot(range(objs.shape[0]), objs[:, 1])
    axes[1].set_title('ect')

    axes[2].plot(range(objs.shape[0]), objs[:, 2])
    axes[2].set_title('ttd')
    plt.show()


def get_data(path):
    if not os.path.exists(path):
        raise Exception(f'path{path} is not existed')
    files = os.listdir(path)
    train_data = []
    for file in files[:]:
        fpath = '/'.join([path, file])
        if os.path.isdir(fpath):
            json_f = os.listdir(fpath)
            for jf in json_f:
                jf_path = '/'.join([fpath, jf])
                train_data.append(jf_path)
        else:
            train_data.append(fpath)
    return train_data


def train():
    args = config()
    args.sa_state_dim = 15
    args.ra_state_dim = 15
    args.train_data = "./data/train/"
    train_data = get_data(args.train_data)
    train_data_size = len(train_data)
    weight, size = init_weight(args.weight_size, args.objective, low_bound=0.1)
    weight = weight[20, :].reshape(1, -1)
    print(weight)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)
    # sa_rlist = []
    # ra_rlist = []
    args.episodes = 200
    objs = np.zeros((args.episodes, args.objective))
    for episode in range(args.episodes):
        # data = train_data[episode % train_data_size]
        data = train_data[1]
        step = 0
        done2 = False
        env = PPO_ENV(data, args)
        cweight(weight, env, sa, ra)
        sa_state, mach_index1, t = env.reset(ra)
        # r1 = 0
        # r2 = 0
        a1 = []
        a2 = []
        while True:
            if env.check_njob_arrival(t):
                job_index = env.njob_insert()
                env.njob_route(job_index, t, ra)

            sa_action = sa.choose_action(sa_state)  # 在mach的候选buffer中选择1个工件进行加工
            a1.append(sa_action)
            job_index, sa_reward, done1, end = env.sa_step(mach_index1, sa_action, t)

            if not done2 and env.jobs[job_index].not_finish():
                ra_state = env.get_ra_state(job_index)
                ra_action = ra.choose_action(ra_state)
                a2.append(ra_action)
                ra_reward, done2 = env.ra_step(job_index, ra_action, t)
                ra.store(ra_state, ra_action, ra_reward, done2)
                # r2 += np.dot(env.w2, np.array(ra_reward))

            sa_state_, mach_index1, t = env.step(ra, t)
            sa.buffer.store(sa_state, sa_action, sa_reward, sa_state_, done1)
            # r1 += np.dot(env.w1, np.array(sa_reward))

            if sa.buffer.cnt == args.batch_size:
                sa.learn(sa_state_, done1)
                # cweight(weight, env, sa=sa, mode=1)
            elif done1 and sa.buffer.cnt != 0:
                sa.learn(sa_state_, done1)

            if ra.buffer.cnt == args.batch_size:
                ra.learn(ra_state, done2)
                # cweight(weight, env, ra=ra, mode=2)
            elif done1 and ra.buffer.cnt != 0:
                ra.learn(ra_state, done2)

            sa_state = sa_state_
            step += 1

            if done1:
                break
        obj = env.cal_objective()
        objs[episode, :] = obj
        print(objs[episode, :].tolist())
        if episode % 10 == 0:
            print(f'a1 = {a1}\na2 = {a2}')
        # print(f'episode{episode} | sa_reward = {r1}, ra_reward = {r2}')
        # sa_rlist.append(r1)
        # ra_rlist.append(r2)
        # if episode % 200 == 0 and episode != 0:
        #    sa.save(f'{episode}epochMix_B{args.batch_size}_W{args.weight_size}_')
        #   ra.save(f'{episode}epochMix_B{args.batch_size}_W{args.weight_size}_')
    show_objective(objs)
    # sa.save(f'{args.episodes}epochMix_B{args.batch_size}_W{args.weight_size}_')
    # ra.save(f'{args.episodes}epochMix_B{args.batch_size}_W{args.weight_size}_')
    # sa.show_loss()
    # ra.show_loss()
    # show_reward(sa_rlist, ra_rlist)


if __name__ == "__main__":
    train()
