# author by 蒋权
import argparse
import os
import json
import time
import numpy as np
from utils.config import config
from utils.rule_env import RULE_ENV
from utils.utils import save_result


def get_action_comb(args):
    action_comb = []
    for i in range(args.sa_action_space):
        for j in range(args.ra_action_space):
            action_comb.append((i, j))
    action_comb = np.array(action_comb)
    np.random.shuffle(action_comb)
    return action_comb.tolist()


def get_random_action(act_comb):
    act_dim = len(act_comb)
    index = np.random.randint(low=0, high=act_dim)
    return act_comb[index]


def change_action(act_comb, env):
    ac = get_random_action(act_comb)
    env.sequence_action = ac[0]
    env.route_action = ac[1]


def random_eval():
    args = config()
    args.test_data = './data/test'
    args.weight_path = './param/pareto_weight/09-19-13-27/weight.npy'
    weight = np.load(args.weight_path)
    size = weight.shape[0]
    instance = ['/'.join([args.test_data, insdir]) for insdir in os.listdir(args.test_data)]
    test_datas = []
    for ins in instance:
        test_datas.append('/'.join([ins, 't0.json']))
    act_comb = get_action_comb(args)

    sa_action = ['FIFO', 'DS', 'EDD', 'CR', 'SRPT']
    ra_action = ['SPT', 'SECM', 'EAM', 'SQT']
    result = {}
    result_dir = './log/eval/multi-random.json'
    for index, data in enumerate(test_datas):
        objs = np.zeros((size, args.objective))
        for i in range(size):
            ac = get_random_action(act_comb)
            env = RULE_ENV(ac, data, args, 5000)
            done2 = False
            mach_index1, t = env.reset()
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t)

                job_index, r, done1, end = env.sa_step(mach_index1, ac[0], t)

                if not done2 and env.jobs[job_index].not_finish():
                    r, done2 = env.ra_step(job_index, ac[1], t)

                mach_index1, t = env.step(None, t)
                if done1:
                    break
                change_action(act_comb, env)
            obj = env.cal_objective()
            objs[i] = obj
            print(f'data {instance[index]} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj3 = {obj[2]}')
        result[instance[index]] = objs.tolist()
    save_result(result, result_dir)
    print('end')


if __name__ == '__main__':
    random_eval()
