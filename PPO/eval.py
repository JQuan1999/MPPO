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
from utils.config import config


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
    args = config()
    args.ra_ckpt_path = ['./param/ra/09-17-15-51/1800epochMix_B128_W100_actor.pkl',
                         './param/ra/09-17-15-51/1800epochMix_B128_W100_critic.pkl']
    args.sa_ckpt_path = ['./param/sa/09-17-15-51/1800epochMix_B128_W100_actor.pkl',
                         './param/sa/09-17-15-51/1800epochMix_B128_W100_critic.pkl']

    args.test_data = './data/test/'
    test = get_data(args.test_data)
    weight, size = init_weight(args.weight_size, args.objective)

    sa = Sequence_Agent(args)
    ra = Route_Agent(args)
    ra.load(args.ra_ckpt_path)
    sa.load(args.sa_ckpt_path)

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
            env = PPO_ENV(test_data[j])
            cweight(weight, env, ra, sa)
            sa_state, mach_index1, t = env.reset(ra)
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t, ra)

                sa_action = sa.choose_action(sa_state)
                a1.append(sa_action)
                job_index, sa_reward, done1, end = env.sa_step(mach_index1, sa_action, t)

                if not done2 and env.jobs[job_index].not_finish():
                    ra_state = env.get_ra_state(job_index)
                    ra_action = ra.choose_action(ra_state)
                    a2.append(ra_action)
                    ra_reward, done2 = env.ra_step(job_index, ra_action, t)

                # env.render()
                sa_state_, mach_index1, t = env.step(ra, t)
                sa_state = sa_state_
                step += 1
                if step % args.batch_size == 0:
                    cweight(weight, env, sa, ra)
                if done1:
                    break
            print(f'a1 = {a1}\n a2={a2}')
            # env.render(t=100)
            obj_v = env.cal_objective()
            objs[j] = obj_v
            # print(f'data {test_data[j]} obj1 = {obj_v[0]}, obj2 = {obj_v[1]}, obj3 = {obj_v[2]}')
        mean_obj = objs.mean(axis=0).tolist()
        print(f'inst {inst} mean obj = {mean_obj}')
        obj_record[inst] = mean_obj
    save_result(obj_record, args)


if __name__ == "__main__":
    eval_()
