# author by 蒋权
import argparse
import os
import json
import time
import numpy as np
from utils.rule_env import RULE_ENV
from utils.generator_data import NoIndentEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str, default='./data/test', help='test data directory path')
parser.add_argument('--seq_action', type=int, default=5, help='sequence action space')
parser.add_argument('--route_action', type=int, default=4, help='route action space')
parser.add_argument('--log_dir', type=str, default='./log/eval', help='directory path to save eval result')


def get_data(dpath):
    files = os.listdir(dpath)
    test_data = []
    for file in files[:]:
        file_path = '/'.join([dpath, file])
        test_data.append(file_path)
    return test_data


def get_action_comb(args):
    action_comb = []
    for i in range(args.seq_action):
        for j in range(args.route_action):
            action_comb.append((i, j))
    action_comb = np.array(action_comb)
    np.random.shuffle(action_comb)
    return action_comb.tolist()


def save_result(record, args):
    result = json.dumps(record, indent=2, sort_keys=True, cls=NoIndentEncoder)
    dir_path = args.log_dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    name = time.strftime('%m-%d-%H-%M') + '-randRule.json'
    file_path = '/'.join([dir_path, name])
    with open(file_path, 'w') as f:
        f.write(result)


def get_random_action(act_comb):
    act_dim = len(act_comb)
    index = np.random.randint(low=0, high=act_dim)
    return act_comb[index]


def change_action(act_comb, env):
    ac = get_random_action(act_comb)
    env.sequence_action = ac[0]
    env.route_action = ac[1]


def random_eval():
    args = parser.parse_args()
    test = get_data(args.test)
    act_comb = get_action_comb(args)

    sa_action = ['FIFO', 'DS', 'EDD', 'CR', 'SRPT']
    ra_action = ['SPT', 'SECM', 'EAM', 'SQT']
    record = {}
    for i in range(len(test)):
        test_data = get_data(test[i])
        inst = test[i].split('/')[-1]
        record[inst] = {}
        obj = np.zeros((len(test_data), 3))
        for j in range(len(test_data)):
            ac = get_random_action(act_comb)
            env = RULE_ENV(test_data[j], ac[0], ac[1], 5000)
            done2 = False
            mach_index1, t = env.reset()
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t)

                job_index = env.sequence_rule(mach_index1, ac[0], t)
                _, done1, _ = env.sa_step(mach_index1, job_index)

                if not done2 and env.jobs[job_index].not_finish():
                    mach_index2 = env.route_rule(job_index, ac[1])
                    _, done2 = env.ra_step(mach_index2, job_index, t)

                mach_index1, t = env.step(None, t)
                if done1:
                    break
                change_action(act_comb, env)
            obj_v = env.cal_objective()
            obj[j] = np.array(obj_v)
        record[inst] = obj.mean(axis=0).tolist()
        print(f'inst {inst} | obj = {record[inst]} ')
    save_result(record, args)


if __name__ == '__main__':
    random_eval()
