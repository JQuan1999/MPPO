import os
import argparse

import numpy as np
import json
import time

from env import PPO_ENV
from agent import Sequence_Agent, Route_Agent
from utils.uniform_weight import init_weight, cweight
from utils.generator_data import NoIndentEncoder


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=500, help='train episodes')
parser.add_argument('--batch_size', type=int, default=128, help='learning batch size')
parser.add_argument('--a_update_step', type=int, default=10, help='actor learning step')
parser.add_argument('--c_update_step', type=int, default=10, help='critic learning step')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='discount1 reward')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
parser.add_argument('--weight_path', type=str, default='', help='weight path')
parser.add_argument('--objective', type=int, default=3, help='objective size')
parser.add_argument('--sa_state_dim', type=int, default=15, help='sequence agent state dim')
parser.add_argument('--ra_state_dim', type=int, default=15, help='route agent state dim')
parser.add_argument('--sa_action_space', type=int, default=5, help='sequence agent action space')
parser.add_argument('--ra_action_space', type=int, default=4, help='route agent action space')
parser.add_argument('--test', type=str, default='', help='test data directory')
parser.add_argument('--sa_ckpt_path', type=str, default='', help='sequence agent ckpt dir')
parser.add_argument('--ra_ckpt_path', type=str, default='', help='route agent ckpt dir')
parser.add_argument('--log_dir', type=str, default='./log/eval/', help='eval result saved path')


def get_data(dpath):
    if os.path.isdir(dpath):
        files = os.listdir(dpath)
        test_data = []
        for file in files[:]:
            file_path = '/'.join([dpath, file])
            test_data.append(file_path)
        return test_data
    else:
        return [dpath]


def get_ckpt(dpath):
    dir_name = os.listdir(dpath)
    ckpt_path = []
    for i in range(len(dir_name)):
        dir_path = '/'.join([dpath, dir_name[i]])
        ckpt_path.append(['/'.join([dir_path, 'actor.pkl']), '/'.join([dir_path, 'critic.pkl'])])
    return ckpt_path


def eval_():
    args = parser.parse_args()
    args.test = './data/test/j20_m20_n50/t0.json'
    args.sa_ckpt = './param/pareto_weight/09-12-20-56/sa'
    args.ra_ckpt = './param/pareto_weight/09-12-20-56/ra'
    args.weight_path = './param/pareto_weight/09-12-20-56/weight.npy'
    result_dir = os.path.dirname(args.weight_path) + '/' + args.test.split('/')[-2]
    print(args)

    test_data = get_data(args.test)[0]
    weight = np.load(args.weight_path)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)
    sa_ckpt = get_ckpt(args.sa_ckpt)
    ra_ckpt = get_ckpt(args.ra_ckpt)
    objs = np.zeros((weight.shape[0], args.objective))
    for i in range(weight.shape[0]):
        sa.load(sa_ckpt[i])
        ra.load(ra_ckpt[i])
        env = PPO_ENV(test_data)
        w = weight[i].reshape(1, -1)
        cweight(w, env, ra, sa)
        sa_state, mach_index1, t = env.reset(ra)
        done2 = False
        a1 = []
        a2 = []
        while True:
            if env.check_njob_arrival(t):
                job_index = env.njob_insert()
                env.njob_route(job_index, t, ra)

            sa_action = sa.choose_action(sa_state)
            a1.append(sa_action)
            job_index = env.sequence_rule(mach_index1, sa_action, t)
            sa_reward, done1, end = env.sa_step(mach_index1, job_index)

            if not done2 and env.jobs[job_index].not_finish():
                ra_state = env.get_ra_state(job_index)
                ra_action = ra.choose_action(ra_state)
                a2.append(ra_action)
                mach_index2 = env.route_rule(job_index, ra_action)
                ra_reward, done2 = env.ra_step(mach_index2, job_index, t)

            # env.render()
            sa_state_, mach_index1, t = env.step(ra, t)
            sa_state = sa_state_
            if done1:
                break
        # env.render(t=100)
        obj = env.cal_objective()
        objs[i] = obj
        print(f'args.test {args.test} weight {weight[i].reshape(-1, ).tolist()} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
        print(f'a1 = {a1}\n a2 = {a2}')
    print(f'save dir = {result_dir}')
    print(f'obj = {objs.tolist()}')
    np.save(result_dir, objs)
    print('end')


if __name__ == '__main__':
    eval_()