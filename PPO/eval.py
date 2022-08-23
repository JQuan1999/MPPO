# author by 蒋权
import os
import argparse

import numpy as np
import json
import time

from env import PPO_ENV
from agent import Sequence_Agent, Route_Agent
from utils.uniform_weight import init_weight, cweight


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=1000, help='train episodes')
parser.add_argument('--batch_size', type=int, default=128, help='learning batch size')
parser.add_argument('--a_update_step', type=int, default=10, help='actor learning step')
parser.add_argument('--c_update_step', type=int, default=10, help='critic learning step')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='discount reward')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
parser.add_argument('--weight_size', type=int, default=36, help='sample weight')
parser.add_argument('--objective', type=int, default=3, help='objective size')
parser.add_argument('--sa_state_dim', type=int, default=12, help='sequence agent state dim')
parser.add_argument('--ra_state_dim', type=int, default=10, help='route agent state dim')
parser.add_argument('--sa_action_space', type=int, default=5, help='sequence agent action space')
parser.add_argument('--ra_action_space', type=int, default=4, help='route agent action space')


def save_result(obj_record):
    result = json.dumps(obj_record)
    dir_path = './log'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    date = time.strftime('%Y-%m-%d-%H-%M-%S') + '.json'
    file_path = '/'.join([dir_path, date])
    with open(file_path, 'w') as f:
        f.write(result)


def eval_():
    path = './data/train'
    files = os.listdir(path)
    test_data = []
    for file in files[-10:]:
        dir_path = '/'.join([path, file])
        if os.path.exists(dir_path):
            json_f = os.listdir(dir_path)
            for jf in json_f:
                jf_path = '/'.join([dir_path, jf])
                test_data.append(jf_path)
    print(f'test data size :{len(test_data)}')

    ra_pkl = ['./param/ra/2022-08-22-17-03-53/1000epoch_actor.pkl', './param/ra/2022-08-22-17-03-53/1000epoch_critic.pkl']
    sa_pkl = ['./param/sa/2022-08-22-17-03-53/1000epoch_actor.pkl', './param/sa/2022-08-22-17-03-53/1000epoch_critic.pkl']
    args = parser.parse_args()
    weight, size = init_weight(args.weight_size, args.objective)
    sa = Sequence_Agent(args.sa_state_dim + args.objective, args.sa_action_space, args)
    ra = Route_Agent(args.ra_state_dim + args.objective, args.ra_action_space, args)
    ra.load(ra_pkl)
    sa.load(sa_pkl)
    obj_record = {}
    for i in range(len(test_data)):
        step = 0
        done2 = False
        env = PPO_ENV(test_data[i])
        w = weight[np.random.randint(low=0, size=size)]
        cweight(w, env, ra, sa)
        sa_state, mach_index1, t = env.reset(ra)
        while True:
            if env.check_njob_arrival(t):
                job_index = env.njob_insert()
                env.njob_route(job_index, ra, t)

            sa_action = sa.choose_action(sa_state)
            job_index = env.sequence_rule(mach_index1, sa_action, t)
            sa_reward, done1, end = env.sa_step(mach_index1, job_index)

            if not done2 and env.jobs[job_index].not_finish():
                ra_state = env.get_ra_state(job_index, end)
                ra_action = ra.choose_action(ra_state)
                mach_index2 = env.route_rule(job_index, ra_action)
                ra_reward, done2 = env.ra_step(mach_index2, job_index, t)

            env.render()
            sa_state_, mach_index1, t = env.step(ra, t)
            sa_state = sa_state_
            step += 1
            if done1:
                break
        obj_v = env.cal_objective()
        obj_record[test_data[i]] = obj_v
        print(f'data {test_data[i]} obj1 = {obj_v[0]}, obj2 = {obj_v[1]}, obj3 = {obj_v[2]}')
    save_result(obj_record)


if __name__ == "__main__":
    eval_()