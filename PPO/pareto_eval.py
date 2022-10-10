import json
import os
import argparse
import time
import numpy as np

from env import PPO_ENV
from agent import Sequence_Agent, Route_Agent
from utils.uniform_weight import cweight
from utils.config import config
from utils.utils import get_data, save_result, get_ckpt


def eval_():
    args = config()
    args.test_data = './data/test'
    instance = ['/'.join([args.test_data, insdir]) for insdir in os.listdir(args.test_data)]
    test_datas = []
    for ins in instance:
        test_datas.append('/'.join([ins, 't0.json']))

    args.sa_ckpt = './param/pareto_weight/10-10-00-27/sa'
    args.ra_ckpt = './param/pareto_weight/10-10-00-27/ra'
    args.weight_path = './param/pareto_weight/10-10-00-27/weight.npy'
    result_dir = './log/eval/multi-agent.json'
    print(args)

    # test_data = get_data(args.test_data)[0]
    weight = np.load(args.weight_path)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)
    sa_ckpt = get_ckpt(args.sa_ckpt)
    ra_ckpt = get_ckpt(args.ra_ckpt)

    result = {}
    for index, data in enumerate(test_datas):
        objs = np.zeros((weight.shape[0], args.objective))
        for i in range(weight.shape[0]):
            sa.load(sa_ckpt[i])
            ra.load(ra_ckpt[i])
            env = PPO_ENV(data, args)
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
            print(f'data {instance[index]} weight {weight[i].reshape(-1, ).tolist()} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
            # print(f'a1 = {a1}\n a2 = {a2}')
        result[instance[index]] = objs.tolist()
    save_result(result, result_dir)
    print('end')


if __name__ == '__main__':
    eval_()