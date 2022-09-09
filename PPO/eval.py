# author by 蒋权
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
parser.add_argument('--weight_size', type=int, default=100, help='sample weight')
parser.add_argument('--objective', type=int, default=3, help='objective size')
parser.add_argument('--sa_state_dim', type=int, default=16, help='sequence agent state dim')
parser.add_argument('--ra_state_dim', type=int, default=16, help='route agent state dim')
parser.add_argument('--sa_action_space', type=int, default=5, help='sequence agent action space')
parser.add_argument('--ra_action_space', type=int, default=4, help='route agent action space')
parser.add_argument('--test', type=str, default='', help='test data directory')
parser.add_argument('--sa_pkl', type=list, default=[], help='sequence agent param weight file')
parser.add_argument('--ra_pkl', type=list, default=[], help='route agent param weight file')
parser.add_argument('--log_dir', type=str, default='./log/eval/', help='eval result saved path')


def save_result(obj_record, args):
    result = json.dumps(obj_record, indent=2, sort_keys=True, cls=NoIndentEncoder)
    dir_path = args.log_dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    name = time.strftime('%m-%d-%H-%M') + '-agent.json'
    file_path = '/'.join([dir_path, name])
    with open(file_path, 'w') as f:
        f.write(result)


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


def eval_():
    args = parser.parse_args()
    args.ra_pkl = ['./param/ra/09-04-00-57/1800epochMix_B128_W100_actor.pkl', './param/ra/09-04-00-57/1800epochMix_B128_W100_critic.pkl']
    args.sa_pkl = ['./param/sa/09-04-00-57/1800epochMix_B128_W100_actor.pkl', './param/sa/09-04-00-57/1800epochMix_B128_W100_critic.pkl']
    args.test = './data/test/j20_m20_n20/'
    test = get_data(args.test)
    weight, size = init_weight(args.weight_size, args.objective)
    # weight = np.tile(np.array([0.8, 0.1, 0.1]), (10, 1))
    sa = Sequence_Agent(args.sa_state_dim + args.objective, args.sa_action_space, args)
    ra = Route_Agent(args.ra_state_dim + args.objective, args.ra_action_space, args)
    ra.load(args.ra_pkl)
    sa.load(args.sa_pkl)
    obj_record = {}
    for i in range(len(test)):
        inst = test[i].split('/')[-1]
        obj_record[inst] = {}
        test_data = get_data(test[i])
        objs = np.zeros((len(test_data), 3))
        for j in range(len(test_data)):
            a1 = []
            a2 = []
            step = 0
            done2 = False
            env = PPO_ENV(test_data[j], t=1000)
            cweight(weight, env, ra, sa)
            sa_state, mach_index1, t = env.reset(ra)
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t, ra)

                sa_action = sa.choose_action(sa_state, train=False)
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
                step += 1
                if step % 128 == 0:
                     cweight(weight, env, sa, ra)
                if done1:
                    break
            # print(f'a1 = {a1}\n a2={a2}')
            env.render(t=100)
            obj_v = env.cal_objective()
            objs[j] = obj_v
            # print(f'data {test_data[j]} obj1 = {obj_v[0]}, obj2 = {obj_v[1]}, obj3 = {obj_v[2]}')
        mean_obj = objs.mean(axis=0).tolist()
        print(f'inst {inst} mean obj = {mean_obj}')
        obj_record[inst] = mean_obj
    save_result(obj_record, args)


if __name__ == "__main__":
    eval_()