# author by 蒋权
import numpy as np
import os
import uuid
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_job_num', type=int, default=0, help='initial job num in the system')
parser.add_argument('--machine_num', type=int, default=0, help='machine num in the system')
parser.add_argument('--new_job_num', type=int, default=0, help='new job num in the system')
parser.add_argument('--process_time', type=list, default=[20, 50], help='process time interval')
parser.add_argument('--operation_num', type=list, default=[0, 0], help='available operation machine num interval')
parser.add_argument('--available_machine', type=list, default=[0, 0], help='available operate machine for an operation')
parser.add_argument('--urgency_degree', type=list, default=[1, 3], help='job urgency degree')
parser.add_argument('--interval_break', type=int, default=500,
                    help='machine break time exponential distribution parameter')
parser.add_argument('--repair_time', type=int, default=50, help='machine repair time ')
parser.add_argument('--break_cnt', type=int, default=50, help='machine max break times')
parser.add_argument('--interval_arrival', type=list, default=[25, 100], help='new job arrival time interval')
parser.add_argument('--ddt_ratio', type=list, default=[1.2, 1.5], help='due date ratio')


class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
            # print(self._replacement_map[key])
            # print("@@%s@@" % (key,))
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        # print("-----------------------------------------------")
        # print("encode被调用")
        result = super(NoIndentEncoder, self).encode(o)
        for k, v in iter(self._replacement_map.items()):
            result = result.replace('"@@%s@@"' % (k,), v)
            # print(result)
        return result


def energy_cost(process_time):
    cost = np.round(0.5 + 40 - 0.5 * process_time + np.random.randint(low=1, high=5, size=process_time.shape))
    return cost


# 工件信息包含:操作数,每个操作的可用机器数,每个操作的具体可用机器编号,每个操作的加工时间,每个操作的加工能耗,紧急程度,截止日期程度
def generator_job_info(data, args, job_key, job_format, arrival_t, flag=False):
    if not flag:
        job_num = args.init_job_num
    else:
        job_num = args.new_job_num
    for j in range(job_num):
        # 操作数
        op_num = np.random.randint(low=args.operation_num[0], high=args.operation_num[1])
        # 可用机器数
        ava_mach_nums = np.random.randint(low=args.available_machine[0], high=args.available_machine[1],
                                          size=op_num).tolist()
        # 具体的可用机器编号
        ava_mach_numbers = []
        for a in ava_mach_nums:
            a_id = np.sort(np.random.choice(args.available_machine[1], a, replace=False)).tolist()
            ava_mach_numbers.append(a_id)
        # 加工时间矩阵
        p_ts = []
        # 能源消耗矩阵
        e_costs = []
        for a in ava_mach_nums:
            # 矩阵大小为op_num * AVAILABLE_MACHINE
            p_t = np.random.randint(low=args.process_time[0], high=args.process_time[1], size=a)
            e_c = energy_cost(p_t).tolist()
            p_t = p_t.tolist()
            p_ts.append(p_t)
            e_costs.append(e_c)
        # 紧急程度
        u_degree = np.random.randint(low=args.urgency_degree[0], high=args.urgency_degree[1])
        # 预期程度
        ddt_ratio = np.random.uniform(low=args.ddt_ratio[0], high=args.ddt_ratio[1])
        name = job_format.format(j)
        data[job_key][name] = {}
        data[job_key][name]["op_num"] = op_num
        data[job_key][name]["ava_mach_nums"] = NoIndent(ava_mach_nums)
        data[job_key][name]["ava_mach_numbers"] = NoIndent(ava_mach_numbers)
        data[job_key][name]["p_ts"] = NoIndent(p_ts)
        data[job_key][name]["e_costs"] = NoIndent(e_costs)
        data[job_key][name]["u_degree"] = u_degree
        data[job_key][name]["ddt_ratio"] = ddt_ratio
        data[job_key][name]["arrival_t"] = arrival_t[j]


# 加工机器的信息包含:机器故障事件间隔,机器故障修理时间
def generator_machine_info(data, args, machine_key, machine_format):
    for m in range(args.machine_num):
        # 机器故障时间服从指数分布 均值 = INTERVAL_FAILURE
        # 生成机器故障的时间节点
        fail_t = np.cumsum(np.round(np.random.exponential(args.interval_break, size=args.break_cnt))).tolist()
        # 机器故障的修复时间
        rep_t = np.round(np.random.exponential(args.repair_time, size=args.break_cnt)).tolist()
        name = machine_format.format(m)
        data[machine_key][name] = {}
        data[machine_key][name]["break_t"] = NoIndent(fail_t)
        data[machine_key][name]["rep_t"] = NoIndent(rep_t)


# 生成插入工件的信息:包含基本工件信息和插入时间
def generator_insert_job_info(data, args, new_job_key, new_job_format):
    mean_at = np.random.randint(low=args.interval_arrival[0], high=args.interval_arrival[1])
    # 工件的到达时间服从指数分布
    # 生成到达时间
    arrival_t = np.cumsum(np.round(np.random.exponential(mean_at, size=args.new_job_num))).tolist()
    generator_job_info(data, args, new_job_key, new_job_format, arrival_t, flag=True)


def generator_data():
    args = parser.parse_args()
    path = "../data/test"

    jn = [10, 10, 10, 20, 20, 20, 30, 30, 30]  # 初始工件数量
    mn = [10, 10, 10, 20, 20, 20, 30, 30, 30]  # 加工机器数量
    nn = [20, 50, 100, 20, 50, 100, 20, 50, 100]  # 新工件数量
    times = 20
    machine_key = "machine"
    machine_format = "machine{}"
    job_key = "job"
    job_format = "job{}"
    new_job_key = "new_job"
    new_job_format = "new_job{}"
    for i in range(len(jn)):
        j = jn[i]
        m = mn[i]
        n = nn[i]
        dire = 'j{}_m{}_n{}'.format(j, m, n)
        data_path = '/'.join([path, dire])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for t in range(times):
            args.init_job_num = j
            args.machine_num = m
            args.new_job_num = n
            args.operation_num = [args.machine_num // 2, args.machine_num]
            args.available_machine = [1, args.machine_num]

            data = {machine_key: {}, job_key: {}, new_job_key: {}}

            data[job_key]["init_job_num"] = j
            data[machine_key]["machine_num"] = m
            data[new_job_key]["new_job_num"] = n

            generator_job_info(data, args, job_key, job_format, np.zeros(args.init_job_num))
            generator_machine_info(data, args, machine_key, machine_format)
            generator_insert_job_info(data, args, new_job_key, new_job_format)

            name = "t{}.json".format(t)
            file = "/".join([data_path, name])
            with open(file, 'w') as f:
                obj = json.dumps(data, indent=2, sort_keys=True, cls=NoIndentEncoder)
                f.write(obj)
    print('finish generating data!!!')


def generate_single():
    args = parser.parse_args()
    path = '../data/train'
    machine_key = "machine"
    machine_format = "machine{}"
    job_key = "job"
    job_format = "job{}"
    new_job_key = "new_job"
    new_job_format = "new_job{}"
    data = {machine_key: {}, job_key: {}, new_job_key: {}}

    args.init_job_num = 10
    args.machine_num = 10
    args.new_job_num = 10
    args.operation_num = [args.machine_num // 2, args.machine_num]
    args.available_machine = [1, args.machine_num]

    data[machine_key]["machine_num"] = args.machine_num
    data[job_key]["init_job_num"] = args.init_job_num
    data[new_job_key]["new_job_num"] = args.new_job_num

    generator_job_info(data, args, job_key, job_format, np.zeros(args.init_job_num))
    generator_machine_info(data, args, machine_key, machine_format)
    generator_insert_job_info(data, args, new_job_key, new_job_format)

    name = 'j{}_m{}_n{}.json'.format(args.init_job_num, args.machine_num, args.new_job_num)
    file = "/".join([path, name])
    with open(file, 'w') as f:
        obj = json.dumps(data, indent=2, sort_keys=True, cls=NoIndentEncoder)
        f.write(obj)


# 生成特定组合的数据
def generate_spec_data():
    args = parser.parse_args()
    path = '../data/train'
    machine_key = "machine"
    machine_format = "machine{}"
    job_key = "job"
    job_format = "job{}"
    new_job_key = "new_job"
    new_job_format = "new_job{}"

    jn = [20]  # 初始工件数量
    mn = [20]  # 加工机器数量
    nn = [50]  # 新工件数量
    times = 100
    for k in range(len(jn)):
        j = jn[k]
        m = mn[k]
        n = nn[k]
        dire = 'j{}_m{}_n{}'.format(j, m, n)
        data_path = '/'.join([path, dire])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for t in range(times):
            args.init_job_num = j
            args.machine_num = m
            args.new_job_num = n
            args.operation_num = [args.machine_num // 2, args.machine_num]
            args.available_machine = [1, args.machine_num]

            data = {machine_key: {}, job_key: {}, new_job_key: {}}

            data[job_key]["init_job_num"] = j
            data[machine_key]["machine_num"] = m
            data[new_job_key]["new_job_num"] = n

            generator_job_info(data, args, job_key, job_format, np.zeros(args.init_job_num))
            generator_machine_info(data, args, machine_key, machine_format)
            generator_insert_job_info(data, args, new_job_key, new_job_format)

            name = "t{}.json".format(t)
            file = "/".join([data_path, name])
            with open(file, 'w') as f:
                obj = json.dumps(data, indent=2, sort_keys=True, cls=NoIndentEncoder)
                f.write(obj)
    print('generate data finished!!!')


if __name__ == "__main__":
    generate_spec_data()
