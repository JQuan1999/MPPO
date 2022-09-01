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
    name = time.strftime('%m-%d-%H-%M') + '-rule.json'
    file_path = '/'.join([dir_path, name])
    with open(file_path, 'w') as f:
        f.write(result)


def rule_eval():
    args = parser.parse_args()
    test = get_data(args.test)
    # test = ['./data/test/j10_m10_n20', './data/test/j10_m10_n50', './data/test/j20_m20_n50', './data/test/j20_m20_n100', './data/test/j30_m30_n50', './data/test/j30_m30_n100']
    act_comb = get_action_comb(args)[:10]
    sa_action = ['DS', 'EDD', 'CR', 'SPT', 'SRPT']
    ra_action = ['SPM', 'SECM', 'EAM', 'SQT']
    record = {}
    for k, ac in enumerate(act_comb):
        comb_name = '_'.join([sa_action[ac[0]], ra_action[ac[1]]])
        record[comb_name] = {}
        for i in range(len(test)):
            test_data = get_data(test[i])
            inst = test[i].split('/')[-1]
            record[comb_name][inst] = {}
            obj = np.zeros((len(test_data), 3))
            for j in range(len(test_data)):
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
                # env.render(1)
                obj_v = env.cal_objective()
                # print(f'data {test_data[j]} obj1 = {obj_v[0]}, obj2 = {obj_v[1]}, obj3 = {obj_v[2]}')
                obj[j] = np.array(obj_v)
            record[comb_name][inst] = obj.mean(axis=0).tolist()
            print(f'{comb_name} | inst{inst} : mean {obj.mean(axis=0).tolist()}')
    save_result(record, args)


if __name__ == '__main__':
    rule_eval()
