import os
import argparse

import numpy as np

from env import PPO_ENV
from agent import Sequence_Agent, Route_Agent
from utils.uniform_weight import cweight
from utils.config import config


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
    args = config()
    args.test_data = './data/test/j20_m20_n50/t0.json'
    args.sa_ckpt = './param/pareto_weight/09-19-13-27/sa'
    args.ra_ckpt = './param/pareto_weight/09-19-13-27/ra'
    args.weight_path = './param/pareto_weight/09-19-13-27/weight.npy'
    result_dir = os.path.dirname(args.weight_path) + '/' + args.test_data.split('/')[-2]
    print(args)

    test_data = get_data(args.test_data)[0]
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
            job_index, sa_reward, done1, end = env.sa_step(mach_index1, sa_action, t)

            if not done2 and env.jobs[job_index].not_finish():
                ra_state = env.get_ra_state(job_index)
                ra_action = ra.choose_action(ra_state)
                a2.append(ra_action)
                ra_reward, done2 = env.ra_step(job_index, ra_action, t)

            # env.render()
            sa_state_, mach_index1, t = env.step(ra, t)
            sa_state = sa_state_
            if done1:
                break
        # env.render(t=100)
        obj = env.cal_objective()
        objs[i] = obj
        print(f'args.test {args.test_data} weight {weight[i].reshape(-1, ).tolist()} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
        print(f'a1 = {a1}\n a2 = {a2}')
    print(f'save dir = {result_dir}')
    print(f'obj = {objs.tolist()}')
    np.save(result_dir, objs)
    print('end')


if __name__ == '__main__':
    eval_()