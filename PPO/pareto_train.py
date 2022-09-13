import argparse
import numpy as np
import os
import time

from agent import Sequence_Agent, Route_Agent
from env import PPO_ENV
from utils.uniform_weight import init_weight, cweight


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='every weight learn epochs')
parser.add_argument('--batch_size', type=int, default=128, help='learning batch size')
parser.add_argument('--a_update_step', type=int, default=10, help='actor learning step')
parser.add_argument('--c_update_step', type=int, default=10, help='critic learning step')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='discount1 reward')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
parser.add_argument('--weight_size', type=int, default=100, help='sample weight')
parser.add_argument('--objective', type=int, default=3, help='objective size')
parser.add_argument('--sa_state_dim', type=int, default=15, help='sequence agent state dim')
parser.add_argument('--ra_state_dim', type=int, default=15, help='route agent state dim')
parser.add_argument('--sa_action_space', type=int, default=5, help='sequence agent action space')
parser.add_argument('--ra_action_space', type=int, default=4, help='route agent action space')
parser.add_argument('--sa_ckpt_path', type=str, default='./param/pareto_weight', help='path to save sa ckpt')
parser.add_argument('--ra_ckpt_path', type=str, default='./param/pareto_weight', help='path to save ra ckpt')


def get_data(path):
    files = os.listdir(path)
    train_data = []
    for file in files[:]:
        dir_path = '/'.join([path, file])
        if os.path.exists(dir_path):
            json_f = os.listdir(dir_path)
            for jf in json_f:
                jf_path = '/'.join([dir_path, jf])
                train_data.append(jf_path)
    return train_data


def train():
    tdata_path = './data/train'
    train_data = get_data(tdata_path)
    train_data_size = len(train_data)

    args = parser.parse_args()
    date = time.strftime('%m-%d-%H-%M')
    args.sa_ckpt_path = '/'.join([args.sa_ckpt_path, date, 'sa'])
    args.ra_ckpt_path = '/'.join([args.ra_ckpt_path, date, 'ra'])

    print(args)

    weight, size = init_weight(args.weight_size, args.objective)
    dir_name = os.path.dirname(args.sa_ckpt_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    weight_path = dir_name + '/' + 'weight.npy'
    np.save(weight_path, weight)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)
    for i in range(size):
        w = weight[i].reshape(1, -1)
        for epoch in range(args.epochs):
            step = 0
            done2 = False
            inst = train_data[(i * args.epochs + epoch) % train_data_size]
            env = PPO_ENV(inst)
            cweight(w, env, sa, ra)
            sa_state, mach_index1, t = env.reset(ra)
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t, ra)

                sa_action = sa.choose_action(sa_state)  # 在mach的候选buffer中选择1个工件进行加工
                job_index = env.sequence_rule(mach_index1, sa_action, t)
                sa_reward, done1, end = env.sa_step(mach_index1, job_index)

                if not done2 and env.jobs[job_index].not_finish():
                    ra_state = env.get_ra_state(job_index)
                    ra_action = ra.choose_action(ra_state)
                    mach_index2 = env.route_rule(job_index, ra_action)
                    ra_reward, done2 = env.ra_step(mach_index2, job_index, t)
                    ra.store(ra_state, ra_action, ra_reward, done2)

                sa_state_, mach_index1, t = env.step(ra, t)
                sa.buffer.store(sa_state, sa_action, sa_reward, sa_state_, done1)
                if sa.buffer.cnt == args.batch_size or ra.buffer.cnt == args.batch_size:
                    if sa.buffer.cnt != 0:
                        sa.learn(sa_state_, done1)
                    if ra.buffer.cnt != 0:
                        ra.learn(ra_state, done2)
                sa_state = sa_state_
                step += 1

                if done1:
                    if sa.buffer.cnt != 0:
                        sa.learn(sa_state_, done1)
                    if ra.buffer.cnt != 0:
                        ra.learn(ra_state, done2)
                    break
            obj = env.cal_objective()
            inst = inst.split('/')[-2]
            print(f'inst {inst}, weight{w.reshape(-1, ).tolist()}, epoch{epoch} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
        sa.save(weight=w)
        ra.save(weight=w)
    sa.show_loss()
    ra.show_loss()


if __name__ == "__main__":
    train()
