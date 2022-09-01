# author by 蒋权
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from agent import Sequence_Agent, Route_Agent
from env import PPO_ENV
from utils.uniform_weight import init_weight, cweight


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=500, help='train episodes')
parser.add_argument('--batch_size', type=int, default=128, help='learning batch size')
parser.add_argument('--a_update_step', type=int, default=10, help='actor learning step')
parser.add_argument('--c_update_step', type=int, default=10, help='critic learning step')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='discount reward')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
parser.add_argument('--weight_size', type=int, default=200, help='sample weight')
parser.add_argument('--objective', type=int, default=3, help='objective size')
parser.add_argument('--sa_state_dim', type=int, default=13, help='sequence agent state dim')
parser.add_argument('--ra_state_dim', type=int, default=11, help='route agent state dim')
parser.add_argument('--sa_action_space', type=int, default=5, help='sequence agent action space')
parser.add_argument('--ra_action_space', type=int, default=4, help='route agent action space')


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


def train():
    path = './data/train'
    files = os.listdir(path)
    train_data = []
    for file in files[:]:
        dir_path = '/'.join([path, file])
        if os.path.exists(dir_path):
            json_f = os.listdir(dir_path)
            for jf in json_f:
                jf_path = '/'.join([dir_path, jf])
                train_data.append(jf_path)
    train_data = train_data[50:150]
    train_data_size = len(train_data)
    print(train_data_size)

    args = parser.parse_args()
    weight, size = init_weight(args.weight_size, args.objective)
    sa = Sequence_Agent(args.sa_state_dim + args.objective, args.sa_action_space, args)
    ra = Route_Agent(args.ra_state_dim + args.objective, args.ra_action_space, args)
    sa_rlist = []
    ra_rlist = []
    for episode in range(args.episodes):
        data = train_data[episode % train_data_size]
        # env.render()
        step = 0
        done2 = False
        env = PPO_ENV(data)
        w = cweight(weight, env, sa, ra)
        sa_state, mach_index1, t = env.reset(ra)
        r1 = 0
        a1 = []
        a2 = []
        r2 = 0
        while True:
            if env.check_njob_arrival(t):
                job_index = env.njob_insert()
                env.njob_route(job_index, t, ra)

            sa_action = sa.choose_action(sa_state)  # 在mach的候选buffer中选择1个工件进行加工
            a1.append(sa_action)
            job_index = env.sequence_rule(mach_index1, sa_action, t)
            sa_reward, done1, end = env.sa_step(mach_index1, job_index)

            if not done2 and env.jobs[job_index].not_finish():
                ra_state = env.get_ra_state(job_index, end)
                ra_action = ra.choose_action(ra_state)
                a2.append(ra_action)
                mach_index2 = env.route_rule(job_index, ra_action)
                ra_reward, done2 = env.ra_step(mach_index2, job_index, t)
                ra.store(ra_state, ra_action, ra_reward, done2)
                r2 += np.dot(env.w, np.array(ra_reward))
            # env.render()
            sa_state_, mach_index1, t = env.step(ra, t)
            sa.buffer.store(sa_state, sa_action, sa_reward, sa_state_, done1)
            r1 += np.dot(env.w, np.array(sa_reward))
            if sa.buffer.cnt == args.batch_size or ra.buffer.cnt == args.batch_size:
                if sa.buffer.cnt != 0:
                    sa_actor_loss, sa_critic_loss = sa.learn(sa_state_, done1)
                    print(f'step {sa.learn_step},sa actor_loss = {sa_actor_loss}, sa critic_loss = {sa_critic_loss}')
                if ra.buffer.cnt != 0:
                    ra_actor_loss, ra_critic_loss = ra.learn(ra_state, done2)
                    print(f'step {ra.learn_step},ra actor_loss = {ra_actor_loss}, ra critic_loss = {ra_critic_loss}')
                cweight(weight, env, sa, ra)
            sa_state = sa_state_
            step += 1

            if done1:
                if sa.buffer.cnt != 0:
                    sa.learn(sa_state_, done1)
                if ra.buffer.cnt != 0:
                    ra.learn(ra_state, done2)
                break
        # print(f'episode{episode} | sa_reward = {r1/step}, ra_reward = {r1/step}')
        sa_rlist.append(r1)
        ra_rlist.append(r2)
        a1 = np.array(a1)
        a2 = np.array(a2)
    sa.show_loss()
    ra.show_loss()
    show_reward(sa_rlist, ra_rlist)
    sa.save(f'{args.episodes}epochMix_B{args.batch_size}_W200_')
    ra.save(f'{args.episodes}epochMix_B{args.batch_size}_W200_')


if __name__ == "__main__":
    train()
