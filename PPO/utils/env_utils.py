# author by 蒋权
import json
import numpy as np


class Op:
    def __init__(self, op_index, job_index, ava_mach_num, ava_mach_number, e_cost, p_t):
        """
        :param op_index: 工序索引
        :param job_index: 所属工件索引
        :param ava_mach_num: 可用机器数量
        :param ava_mach_number: 可用机器编号
        :param e_cost: 能耗
        :param p_t: 所需时间
        """
        self.op_index = op_index
        self.job_index = job_index
        self.ava_mach_num = ava_mach_num
        self.ava_mach_number = ava_mach_number
        self.ect = np.array(e_cost)
        self.pt = np.array(p_t)
        self.start = 0
        self.end = 0
        self.f_ect = 0
        self.f_pt = 0
        self.choose_mach_index = None
        self._cal()

    def _cal(self):
        self.ave_pt = np.round(np.sum(self.pt) / len(self.pt))
        self.ave_ect = np.round(np.sum(self.ect) / len(self.ect))

    def get_pt(self, mach_index):
        index = np.where(np.array(self.ava_mach_number) == mach_index)[0]
        if len(index) == 0:
            raise Exception("mach_index {} is not in ava_mach_number" % mach_index)
        return self.pt[index[0]]

    def get_ect(self, mach_index):
        index = np.where(np.array(self.ava_mach_number) == mach_index)[0]
        if len(index) == 0:
            raise Exception("mach_index {} is not in ava_mach_number" % mach_index)
        return self.ect[index[0]]

    def get_ep_rate(self):
        ep_rate = self.ect / self.pt
        return ep_rate

    def sa_step(self, start, end):
        self.start = start
        self.end = end

    def ra_step(self, pt, ect, mach_index):
        self.f_pt = pt
        self.f_ect = ect
        self.choose_mach_index = mach_index


class Job:
    def __init__(self, ops, ddt_ratio, ops_num, u_degree, arrival_t):
        """
        :param ops: 工序集合
        :param ddt: 预期时间
        :param ops_num: 工序数量
        :param u_degree: 紧急程度
        :param arrival_t: 到达时间
        """
        self.ops = ops
        self.ddt_ratio = ddt_ratio
        self.ops_num = ops_num
        self.u_degree = u_degree

        self.pre_op = self.ops[0]  # 当前工序
        self.pre_no = 0  # 当前工序编号
        self.arrival_t = arrival_t
        self.pre_start = self.arrival_t  # 当前工序的开始时间

        self.ops_dispatch_no = 0  # 分配工序编号
        self.ops_finish = np.zeros(self.ops_num).astype(bool)  # 是否完成
        self.ops_dispatch_flag = np.zeros(self.ops_num).astype(bool)  # 是否分配
        self.ops_start = np.zeros(self.ops_num)  # 时间记录
        self.ops_end = np.zeros(self.ops_num)
        self.ops_ects = np.zeros(self.ops_num)  # 能耗记录
        self._cal()

    def _cal(self):
        ops_ave_pt = []  # 工序的平均处理时间
        ops_ave_ect = []  # 工序的平均能耗
        for op in self.ops:
            ops_ave_pt.append(op.ave_pt)
            ops_ave_ect.append(op.ave_ect)
        self.ops_ave_pt = np.array(ops_ave_pt)
        self.ops_ave_ect = np.array(ops_ave_ect)
        self.due_date = round(self.ddt_ratio * np.sum(self.ops_ave_pt)) + self.arrival_t

    def get_tardiness(self, t, flag=False):
        if self.ops_finish.sum() == self.ops_num:
            slack = self.due_date - self.ops_end[-1]
        else:
            # flag = False 表示计算实际延迟时间 flag = True计算预估延迟时间
            slack = self.get_slack_time(t, flag)
        if slack < 0:
            return - self.u_degree * slack
        else:
            return 0

    def get_slack_time(self, t, flag=False):
        left_pt = self.get_remain_pt(flag)
        slack = self.due_date - t - left_pt
        return slack

    def get_remain_pt(self, flag=False):
        if flag is True and self.ops_dispatch_no == self.pre_no + 1:
            left_pt = 0
        else:
            left_pt = self.pre_op.ave_pt
        for op_index in range(self.pre_no + 1, self.ops_num):
            left_pt += self.ops[op_index].ave_pt
        return left_pt

    def get_finish_rate(self):
        finish_num = self.ops_finish.sum()
        return finish_num / self.ops_num

    def sa_step(self, start, end):
        self.ops_start[self.pre_no] = start
        self.ops_end[self.pre_no] = end
        self.pre_start = end
        self.ops_finish[self.pre_no] = True
        self.pre_no += 1
        if self.pre_no != self.ops_num:
            self.pre_op = self.ops[self.pre_no]

    def ra_step(self, ect):
        self.ops_ects[self.pre_no] = ect
        if self.ops_dispatch_no >= self.ops_num:
            raise Exception('all op has been dispatched')
        self.ops_dispatch_flag[self.ops_dispatch_no] = True
        self.ops_dispatch_no += 1

    def cal_ave_ect(self):
        return np.array(self.ops_ects).mean()

    def not_finish(self):
        if self.pre_no == self.ops_num:
            return False
        else:
            return True


class Machine:
    def __init__(self, mach_index, break_t, rep_t):
        self.mach_index = mach_index
        self.break_t = break_t
        self.rep_t = rep_t
        self.break_cnt = 0
        self.op_start = []  # 所有工序的开始时间
        self.op_pt = []  # 所有工序所需处理时间
        self.op_end = []
        self.ects = []
        self.pre_op = None
        self.finished_op = []  # 已完成工序
        self.buffer_op = []  # 队列中工序
        self.ava_t = 0
        self.last_est_ur = 0  # 预计利用率
        self.last_ur = 0  # 利用率
        self.last_ep_ratio = 0  # 能耗加工时间比
        self.idle_cost = 0.2
        self.total_ect = 0
        self.est_ep_ratio = 0

    def _swap(self, index1, index2, l):
        temp = l[index1]
        l[index1] = l[index2]
        l[index2] = temp

    def change(self, index1, index2):
        self._swap(index1, index2, self.op_start)
        self._swap(index1, index2, self.op_end)
        self._swap(index1, index2, self.op_pt)
        self._swap(index1, index2, self.ects)

    def sa_step(self, job_number):
        job_inds = np.array(self.get_queue_job_index())
        b_index = np.where(job_inds == job_number)[0][0]  # buffer中op的index
        begin_index = len(self.finished_op)  # buffer中工件在记录信息中的开始索引
        index = begin_index + b_index  # 记录信息的索引
        break_t, rep_t = self.get_break()  # 获取机器故障时间点

        start = max(self.ava_t, self.op_start[index])
        end = start + self.op_pt[index]
        if break_t <= start < break_t + rep_t:  # 开始时间位于故障开始时间点和修复时间点之间
            start = break_t + rep_t
            # print(f'machine {self.mach_index} the {self.break_cnt}th break time is start: {start} and rep{rep_t}')
            self.break_cnt += 1
        elif start < break_t < end:  # 故障时间点位于开始和结束之间
            end = end + rep_t
            # print(f'machine {self.mach_index} the {self.break_cnt}th break time is start: {start} and rep{rep_t}')
            self.break_cnt += 1
        elif break_t + rep_t <= start:
            # print(f'machine {self.mach_index} the {self.break_cnt}th break time is start: {start} and rep{rep_t}')
            self.break_cnt += 1
        ect = self.ects[index]
        # 更新信息
        self.op_start[index] = start
        self.op_end[index] = end
        self.ava_t = end

        # 如果b_index不等于0 该工件为插入工件 交换begin_index和begin_index+b_index处记录信息
        if b_index != 0:
            self.change(begin_index, index)
        op = self.buffer_op.pop(b_index)
        self.finished_op.append(op)
        self.pre_op = op
        return op, start, end

    def ra_step(self, op, op_start):
        pt = op.get_pt(self.mach_index)
        ect = op.get_ect(self.mach_index)
        self.buffer_op.append(op)
        self.op_start.append(op_start)
        self.op_pt.append(pt)
        self.op_end.append(op_start + pt)
        self.ects.append(ect)
        return pt, ect

    def cal_work_load(self, end=None):
        if end is None:
            end = len(self.finished_op)
        work_load = 0
        for i in range(end):
            work_load += (self.op_end[i] - self.op_start[i])
        return work_load

    def cal_use_ratio(self, end=None):
        if end is None:
            end = len(self.finished_op)
        if end == 0:
            return 0
        work_load = self.cal_work_load(end)
        index = end - 1
        use_ratio = work_load / self.op_end[index]
        return use_ratio

    def cal_ep_ratio(self):
        sum_ect = np.array(self.ects).sum()
        sum_pt = np.array(self.op_pt).sum()
        return sum_ect / sum_pt

    def get_queue_job_index(self):
        job_inds = [op.job_index for op in self.buffer_op]
        return job_inds

    def get_sa_job_index(self):
        # 找出工件到达时间op_start大于等于self.ava_t的工件
        begin_i = len(self.finished_op)
        ava_index = []
        min_ava_index = begin_i  # 最小开始时间索引
        min_start = self.op_start[min_ava_index]  # 最小开始时间
        if min_start <= self.ava_t:
            ava_index.append(min_ava_index)
        for i in range(begin_i + 1, len(self.op_start)):
            if self.op_start[i] <= self.ava_t:
                ava_index.append(i)
            if self.op_start[i] < min_start:
                min_ava_index = i
                min_start = self.op_start[i]
        # 没有开始时间早于机器可用时间的工件
        if len(ava_index) == 0:
            job_index = self.buffer_op[min_ava_index - begin_i].job_index
            return [job_index]
        else:
            job_inds = [self.buffer_op[i - begin_i].job_index for i in ava_index]  # 返回buffer中早于等于机器可用时间的工件索引
            return job_inds

    def get_queue_job_num(self):
        job_inds = self.get_queue_job_index()
        num = len(job_inds)
        return num

    def get_queue_start_time(self):
        begin_index = len(self.finished_op)
        start_t = []
        for i in range(begin_index, len(self.op_start)):
            start_t.append(self.op_start[i])
        return start_t

    def get_estimate_ava_time(self):
        begin_index = len(self.finished_op)
        ava_t = self.ava_t
        for i in range(begin_index, len(self.op_pt)):
            ava_t = max(ava_t, self.op_start[i])
            ava_t += self.op_pt[i]
        return ava_t

    def get_estimate_use_ratio(self):
        begin_i = len(self.finished_op)
        work_load = self.cal_work_load()
        ava_t = self.ava_t
        end = ava_t
        for i in range(begin_i, len(self.op_start)):
            ava_t = max(end, self.op_start[i])
            end += self.op_pt[i]
            work_load += self.op_pt[i]
        if end == 0:
            est_use_ratio = 0
        else:
            est_use_ratio = work_load / end
        return est_use_ratio

    def get_estimate_ep_ratio(self):
        begin_i = len(self.finished_op)
        work_load = self.cal_work_load()
        ava_t = self.ava_t
        end = ava_t
        for i in range(begin_i, len(self.op_start)):
            ava_t = max(end, self.op_start[i])
            end = ava_t + self.op_pt[i]
            work_load += self.op_pt[i]
        idle_t = end - work_load
        total_ect = work_load + idle_t * self.idle_cost
        if end == 0:
            return 0
        else:
            return total_ect / end

    def get_break(self):
        cnt = self.break_cnt
        break_point = self.break_t[cnt]
        rep_t = self.rep_t[cnt]
        return break_point, rep_t

    def cal_total_ect(self):
        work_ect = np.array(self.ects).sum()
        idle_time = 0
        for i in range(len(self.op_end)-1):
            idle_time += self.op_end[i] - self.op_start[i]
        idle_ect = idle_time * self.idle_cost
        ect = work_ect + idle_ect
        # print(f'word_ect{work_ect}, idle_ect{idle_ect}')
        return ect

    def get_state(self):
        queue_job_pt = np.zeros(len(self.buffer_op))
        job_index = np.zeros(len(self.buffer_op)).astype(int)
        begin = len(self.finished_op)
        start = np.zeros(len(self.buffer_op))
        for op_index in range(begin, len(self.op_pt)):
            queue_job_pt[op_index - begin] = self.op_pt[op_index]
            start[op_index - begin] = self.op_start[op_index]
            job_index[op_index - begin] = self.buffer_op[op_index - begin].job_index
        # sum_pt = queue_job_pt.sum()
        mean_pt = queue_job_pt.mean()
        min_pt = queue_job_pt.min()
        mean_start = start.mean()
        min_start = start.min()
        return self.ava_t, mean_start, min_start, mean_pt, min_pt, job_index
        # return sum_pt, mean_pt, min_pt, job_index

    def ur_update(self, ur):
        self.last_ur = ur

    def ect_ur_update(self, est_ur):
        self.last_est_ur = est_ur

    def ep_ratio_update(self, ep_r):
        self.last_ep_ratio = ep_r

    def total_ect_update(self, ect):
        self.total_ect = ect

    def est_ep_ratio_update(self, r):
        self.est_ep_ratio = r


def get_job_data(data, key, job_format):
    job_info = data[key]
    if key == "job":
        job_num = job_info["init_job_num"]
    else:
        job_num = job_info["new_job_num"]
    jobs = []
    for j in range(job_num):
        name = job_format.format(j)
        ava_mach_numbers = job_info[name]["ava_mach_numbers"]
        ava_mach_nums = job_info[name]["ava_mach_nums"]
        ddt_ratio = job_info[name]["ddt_ratio"]
        e_costs = job_info[name]["e_costs"]
        op_num = job_info[name]["op_num"]
        p_ts = job_info[name]["p_ts"]
        u_degree = job_info[name]["u_degree"]
        arrival_t = job_info[name]["arrival_t"]
        ops = []
        for op_index in range(op_num):
            ops.append(Op(op_index, j, ava_mach_nums[op_index], ava_mach_numbers[op_index],
                          e_costs[op_index], p_ts[op_index]))
        ops = np.array(ops)
        jobs.append(Job(ops, ddt_ratio, op_num, u_degree, arrival_t))
    jobs = np.array(jobs)
    return jobs


def get_machine_data(data, key, machine_format):
    machine_info = data[key]
    machine_num = machine_info['machine_num']
    machines = []
    for m in range(machine_num):
        name = machine_format.format(m)
        fail_t = machine_info[name]["break_t"]
        rep_t = machine_info[name]["rep_t"]
        machines.append(Machine(m, fail_t, rep_t))
    machines = np.array(machines)
    return machines


def get_data(instance):
    with open(instance, 'r') as f:
        data = json.load(f)

    machine_key = "machine"
    machine_format = "machine{}"
    job_key = "job"
    job_format = "job{}"
    new_job_key = "new_job"
    new_job_format = "new_job{}"

    machines = get_machine_data(data, machine_key, machine_format)
    jobs = get_job_data(data, job_key, job_format)
    new_jobs = get_job_data(data, new_job_key, new_job_format)
    return jobs, machines, new_jobs


if __name__ == "__main__":
    get_data("../data/j2_m3_n2.json")
