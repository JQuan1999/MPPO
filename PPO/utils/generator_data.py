# author by 蒋权
import numpy as np
import os
import uuid
import json

MACHINE_NUM = 5
JOB_NUM = 5
NEW_JOB_NUM = 10
PROCESS_TIME = [20, 50]
OPERATION_NUM = [MACHINE_NUM // 2, MACHINE_NUM]
AVAILABLE_MACHINE = [1, MACHINE_NUM]
URGENCY_DEGREE = [1, 4]
INTERVAL_FAILURE = 500
REPAIR_TIME = 50
INTERVAL_ARRIVAL = [25, 100]
DDT = [1.0, 1.5]


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
def generator_job_info(data, job, key, format, arrival_t):
    for j in range(job):
        # 操作数
        op_num = np.random.randint(low=OPERATION_NUM[0], high=OPERATION_NUM[1])
        # 可用机器数
        ava_mach_nums = np.random.randint(low=AVAILABLE_MACHINE[0], high=AVAILABLE_MACHINE[1], size=op_num).tolist()
        # 具体的可用机器编号
        ava_mach_numbers = []
        for a in ava_mach_nums:
            # a_id = np.random.randint(low=0, high=MACHINE_NUM, size=a).tolist()
            a_id = np.sort(np.random.choice(AVAILABLE_MACHINE[1], a, replace=False)).tolist()
            ava_mach_numbers.append(a_id)
        # 加工时间矩阵
        p_ts = []
        # 能源消耗矩阵
        e_costs = []
        for a in ava_mach_nums:
            # 矩阵大小为op_num * AVAILABLE_MACHINE
            p_t = np.random.randint(low=PROCESS_TIME[0], high=PROCESS_TIME[1], size=a)
            e_c = energy_cost(p_t).tolist()
            p_t = p_t.tolist()
            p_ts.append(p_t)
            e_costs.append(e_c)
        # 紧急程度
        u_degree = np.random.randint(low=URGENCY_DEGREE[0], high=URGENCY_DEGREE[1])
        # 预期程度
        ddt = np.random.uniform(low=DDT[0], high=DDT[1])
        name = format.format(j)
        data[key][name] = {}
        data[key][name]["op_num"] = op_num
        data[key][name]["ava_mach_nums"] = NoIndent(ava_mach_nums)
        data[key][name]["ava_mach_numbers"] = NoIndent(ava_mach_numbers)
        data[key][name]["p_ts"] = NoIndent(p_ts)
        data[key][name]["e_costs"] = NoIndent(e_costs)
        data[key][name]["u_degree"] = u_degree
        data[key][name]["ddt"] = ddt
        data[key][name]["arrival_t"] = arrival_t[j]


# 加工机器的信息包含:机器故障事件间隔,机器故障修理时间
def generator_machine_info(data, machine, key, format):
    for m in range(machine):
        # 机器故障时间服从指数分布 均值 = INTERVAL_FAILURE
        # 生成机器故障的时间节点
        fail_t = np.cumsum(np.round(np.random.exponential(INTERVAL_FAILURE, size=100))).tolist()
        # 机器故障的修复时间
        rep_t = np.round(np.random.exponential(REPAIR_TIME, size=100)).tolist()
        name = format.format(m)
        data[key][name] = {}
        data[key][name]["break_t"] = NoIndent(fail_t)
        data[key][name]["rep_t"] = NoIndent(rep_t)


# 生成插入工件的信息:包含基本工件信息和插入时间
def generator_insert_job_info(data, new, key, format):
    mean_at = np.random.randint(low=INTERVAL_ARRIVAL[0], high=INTERVAL_ARRIVAL[1])
    # 工件的到达时间服从指数分布
    # 生成到达时间
    arrival_t = np.cumsum(np.round(np.random.exponential(mean_at, size=new))).tolist()
    generator_job_info(data, new, key, format, arrival_t)


def generator_data():

    path = "../data/train"

    jn = [10, 20, 30] # 初始工件数量
    mn = [10, 20, 30] # 加工机器数量
    nn = [20, 50, 100] # 新工件数量
    times = 50
    machine_key = "machine"
    machine_format = "machine{}"
    job_key = "job"
    job_format = "job{}"
    new_job_key = "new_job"
    new_job_format = "new_job{}"
    for j in jn:
        for m in mn:
            for n in nn:
                dire = 'j{}_m{}_n{}'.format(j, m, n)
                data_path = '/'.join([path, dire])
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                for t in range(times):
                    JOB_NUM = j
                    MACHINE_NUM = m
                    NEW_JOB_NUM = n
                    data = {machine_key: {}, job_key: {}, new_job_key: {}}

                    data[machine_key]["machine_num"] = MACHINE_NUM
                    data[job_key]["job_num"] = JOB_NUM
                    data[new_job_key]["new_job_num"] = NEW_JOB_NUM

                    generator_job_info(data, JOB_NUM, job_key, job_format, np.zeros(JOB_NUM))
                    generator_machine_info(data, MACHINE_NUM, machine_key, machine_format)
                    generator_insert_job_info(data, NEW_JOB_NUM, new_job_key, new_job_format)

                    name = "t{}.json".format(t)
                    file = "/".join([data_path, name])
                    with open(file, 'w') as f:
                        obj = json.dumps(data, indent=2, sort_keys=True, cls=NoIndentEncoder)
                        f.write(obj)


if __name__ == "__main__":
    # generator_data()
    path = '../data'
    machine_key = "machine"
    machine_format = "machine{}"
    job_key = "job"
    job_format = "job{}"
    new_job_key = "new_job"
    new_job_format = "new_job{}"
    data = {machine_key: {}, job_key: {}, new_job_key: {}}

    data[machine_key]["machine_num"] = MACHINE_NUM
    data[job_key]["job_num"] = JOB_NUM
    data[new_job_key]["new_job_num"] = NEW_JOB_NUM

    generator_job_info(data, JOB_NUM, job_key, job_format, np.zeros(JOB_NUM))
    generator_machine_info(data, MACHINE_NUM, machine_key, machine_format)
    generator_insert_job_info(data, NEW_JOB_NUM, new_job_key, new_job_format)

    name = 'j{}_m{}_n{}.json'.format(JOB_NUM, MACHINE_NUM, NEW_JOB_NUM)
    file = "/".join([path, name])
    with open(file, 'w') as f:
        obj = json.dumps(data, indent=2, sort_keys=True, cls=NoIndentEncoder)
        f.write(obj)