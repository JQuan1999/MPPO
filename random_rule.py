# author by 蒋权
import argparse
import os
import json
import time
import numpy as np
from utils.config import config
from utils.rule_env import RULE_ENV
from utils.utils import save_result, get_data


np.random.seed(1)


def _random_eval(sa_action, ra_action, args, result_file):
    assert type(sa_action) is list, f"sa action {sa_action} is not a list"
    assert type(ra_action) is list, f"ra action {ra_action} is not a list"

    def get_comb_action():
        ac1 = sa_action[np.random.randint(len(sa_action))]
        ac2 = ra_action[np.random.randint(len(ra_action))]
        return ac1, ac2

    weight = np.load(args.weight_path)
    size = weight.shape[0]
    test_datas = ['/'.join([args.test_data, insdir, 't0.json']) for insdir in os.listdir(args.test_data)][2:]

    result = {}

    for index, data in enumerate(test_datas):
        objs = np.zeros((size, args.objective))
        data_name = data.split('/')[-2]
        begin = time.time()
        for i in range(size):
            ac = get_comb_action()
            env = RULE_ENV(ac, data, args)
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
                ac = get_comb_action()
                env.sequence_action = ac[0]
                env.route_action = ac[1]
            # env.render(t=10, key=data_name)
            obj = env.cal_objective()
            objs[i] = obj
            print(f'data {data_name} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj3 = {obj[2]}')
        t = time.time() - begin
        result[data_name] = {}
        result[data_name]["time"] = t
        result[data_name]["result"] = objs.tolist()
    save_result(result, result_file)
    print('end')


def random_eval():
    args = config()
    args.weight_path = './param/pareto_weight/10-24-16-53/weight.npy'
    args.test_data = "./data/test4"
    sa_action = [i for i in range(args.sa_action_space)]
    ra_action = [i for i in range(args.sa_action_space)]
    prefix = time.strftime('%m-%d-%H-%M')
    name = prefix + "-multi-random.json"
    result_file = './log/eval/' + name
    _random_eval(sa_action, ra_action, args, result_file)
    print('----------end------------')


def part_random():
    args = config()
    args.weight_path = './param/pareto_weight/10-24-16-53/weight.npy'
    args.test_data = "./data/test4"
    sa_action = [i for i in range(args.sa_action_space)]
    ra_action = [i for i in range(args.sa_action_space)]
    sa_action_name = ['FIFO', 'MS', 'EDD', 'CR']
    ra_action_name = ['SPT', 'SEC', 'EA', 'SQT']
    for i in range(2):
        if i == 0:
            choose_action = sa_action
        else:
            choose_action = ra_action
        for c_a in choose_action:
            if i == 0:
                action_name = sa_action_name[c_a]
            else:
                action_name = ra_action_name[c_a]
            prefix = time.strftime('%m-%d-%H-%M')
            name = prefix + f"-random-{action_name}.json"
            result_file = './log/eval/' + name
            if i == 0:
                _random_eval([c_a], ra_action, args, result_file)
            else:
                _random_eval(sa_action, [c_a], args, result_file)
            print('------------------------')


if __name__ == '__main__':
    random_eval()
