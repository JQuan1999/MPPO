# author by 蒋权
import numpy as np
import matplotlib.pyplot as plt

from utils.env_utils import get_data
from utils.dispatch_rule import job_dispatch, machine_dispatch

color = [0.77, 0.18, 0.78,
         0.21, 0.33, 0.64,
         0.88, 0.17, 0.56,
         0.20, 0.69, 0.28,
         0.26, 0.15, 0.47,
         0.83, 0.27, 0.44,
         0.87, 0.85, 0.42,
         0.85, 0.51, 0.87,
         0.99, 0.62, 0.76,
         0.52, 0.43, 0.87,
         0.00, 0.68, 0.92,
         0.26, 0.45, 0.77,
         0.98, 0.75, 0.00,
         0.72, 0.81, 0.76,
         0.77, 0.18, 0.78,
         0.28, 0.39, 0.44,
         0.22, 0.26, 0.24,
         0.64, 0.52, 0.64,
         0.87, 0.73, 0.78,
         0.94, 0.89, 0.85,
         0.85, 0.84, 0.86]


class PPO_ENV:
    def __init__(self, instance, args, t=5000):
        self.jobs, self.machines, self.new_jobs = get_data(instance)
        self.new_arrival_t = [self.new_jobs[i].arrival_t for i in range(len(self.new_jobs))]
        self.job_num = len(self.jobs)
        self.mch_num = len(self.machines)
        self.new_num = len(self.new_jobs)
        self.sa_state_dim = args.sa_state_dim + args.objective
        self.ra_state_dim = args.ra_state_dim + args.objective
        self.sys_state_dim = 8
        self.w_dim = 3
        self.sa_action_space = 5
        self.ra_action_space = 4
        self.t = t
        self.new_job_cnt = 0
        self.last_tardiness = 0  # 上一轮的平均加权延迟时间
        self.last_use_ratio = 0  # 上一轮机器利用率
        self.last_ep_ratio = 0  # 上一轮ecost与pt比值
        self.last_est_ur = 0  # 上一轮机器预估利用率
        self.last_est_tardiness = 0
        self.sum_pt = 0
        self.sum_ect = 0
        self.discount1 = 0.9
        self.discount2 = 0.9
        self.d_flag = False
        self.state1 = np.zeros(self.sys_state_dim).tolist()  # state1 为sa的上一状态
        self.state2 = np.zeros(self.sys_state_dim).tolist()  # state2 为ra的上一状态
        self.w1 = None
        self.w2 = None
        self.color = np.array(color).reshape(-1, 3).tolist()
        self.routed_flag = np.zeros(len(self.jobs)).astype(bool)
        self.finished_flag = np.zeros(len(self.jobs)).astype(bool)

    def sys_state(self, mach_index=None):
        mach_ur = np.zeros(self.mch_num)
        ects = np.zeros(self.mch_num)
        ava_t = np.zeros(self.mch_num)
        for i in range(self.mch_num):
            if i == mach_index and mach_index is not None:
                mach_ur[i] = self.machines[i].cal_use_ratio()
                self.machines[i].ur_update(mach_ur[i])
                ects[i] = self.machines[i].cal_total_ect()
                self.machines[i].total_ect_update(ects[i])
            else:
                mach_ur[i] = self.machines[i].last_ur
                ects[i] = self.machines[i].total_ect
            ava_t[i] = self.machines[i].ava_t
        mean_ur = mach_ur.mean()
        std_ur = mach_ur.std()
        mean_ect = ects.mean()
        std_ect = ects.std()
        # mean_ect = ects.mean()
        # std_ect = ects.std()
        tcur = ava_t.mean()
        cro = np.zeros(len(self.jobs))
        td = np.zeros(len(self.jobs))
        for i in range(len(cro)):
            cro[i] = self.jobs[i].get_finish_rate()
            td[i] = self.jobs[i].get_tardiness(tcur)
        cro_mean = cro.mean()
        cro_std = cro.std()
        td_mean = td.mean()
        td_sum = td.sum()
        # td_mean = td.mean()
        # td_sum = td.sum()
        state = [mean_ur, std_ur, mean_ect, std_ect, cro_mean, cro_std, td_mean, td_sum]
        return state

    def get_sa_state(self, mach_index):
        state = self.sys_state(mach_index)
        # diff = (np.array(state) - self.state1).tolist()
        # self.state_update(state, mode=1)
        # 机器编号,机器队列工件数目,可用时间,预估可用时间,机器利用率,机器累计能耗,机器延迟时间
        id = float(mach_index)
        num = self.machines[mach_index].get_queue_job_num()
        ava_t = self.machines[mach_index].ava_t
        est_ava_t = self.machines[mach_index].get_estimate_ava_time()
        use_ratio = self.machines[mach_index].cal_use_ratio()
        ect = self.machines[mach_index].cal_total_ect()
        ids = self.machines[mach_index].get_queue_job_index()
        td = 0
        for job_index in ids:
            td += self.jobs[job_index].get_tardiness(ava_t)
        mach_state = [id, num, ava_t, est_ava_t, use_ratio, ect, td]
        sa_state = state + mach_state + self.w1
        # sa_state = state + diff + self.w1
        return sa_state

    def get_ra_state(self, job_index):
        state = self.sys_state()
        diff = (np.array(state) - self.state2).tolist()
        self.state_update(state, mode=2)
        # 工件编号,工件完成率,工件开始时间,延迟时间,剩余时间,当前工序加工时间,工序能耗
        id = float(job_index)
        cr = self.jobs[job_index].get_finish_rate()
        start = self.jobs[job_index].pre_start
        ava_t = np.zeros(len(self.machines))
        for i in range(len(self.machines)):
            m = self.machines[i]
            ava_t[i] = m.ava_t
        tcur = ava_t.mean()
        td = self.jobs[job_index].get_tardiness(tcur)
        remain = self.jobs[job_index].get_remain_pt()
        pt = self.jobs[job_index].pre_op.ave_pt
        ect = self.jobs[job_index].pre_op.ave_ect
        jstate = [id, cr, start, td, remain, pt, ect]
        ra_state = state + jstate + self.w2
        # ra_state = state + diff + self.w2
        return ra_state

    def cal_schedule_time(self, ra):
        init = False
        choose = None
        t = None
        for i in range(len(self.machines)):
            m = self.machines[i]
            if len(m.buffer_op) == 0:
                continue
            if not init:
                choose = i
                t = m.ava_t
                init = True
            else:
                if m.ava_t < t:
                    choose = i
                    t = m.ava_t
        if choose is None:  # 工件的buffer都为空 切换到下一个新工件到达时间点
            if len(self.new_jobs) == 0:
                raise Exception('no job will arrival, simulation should be terminal')
            t = self.new_arrival_t[self.new_job_cnt]
            job_index = self.njob_insert()
            choose = self.njob_route(job_index, t, ra)
        return choose, t

    def ra_done(self, t):
        if self.t <= t:  # 达到预设停止时间
            # print('time terminal')
            return True
        elif self.new_job_cnt == self.new_num and self.routed_flag.sum() == len(self.routed_flag):  # 所有工件都已分配
            # print('no job will arrive')
            return True
        else:
            return False

    def sa_done(self, t):
        if self.t <= t:  # 达到预设停止时间
            return True
        elif self.new_job_cnt == self.new_num and self.finished_flag.sum() == len(self.finished_flag):  # 所有工件都已完成加工
            return True
        else:
            return False

    def state_update(self, s, mode):
        if mode == 1:
            self.state1 = s
        else:
            self.state2 = s

    def sa_reward(self, mach_index):
        ur = np.zeros(len(self.machines))
        ava_t = np.zeros(len(self.machines))
        for i in range(len(self.machines)):
            if i == mach_index:
                ur[i] = self.machines[i].cal_use_ratio()
            else:
                ur[i] = self.machines[i].last_ur
            ava_t[i] = self.machines[i].ava_t
        mean_ur = ur.mean()
        if mean_ur > self.last_use_ratio:
            r1 = 1
        elif mean_ur == self.last_use_ratio:
            r1 = 0
        else:
            r1 = -1
        self.last_use_ratio = mean_ur

        tcur = ava_t.mean()
        tardiness = np.zeros(len(self.jobs))
        for j in range(len(self.jobs)):
            if self.finished_flag[j] is True:
                continue
            tardiness[j] = self.jobs[j].get_tardiness(tcur)
        mean_tardiness = tardiness.mean()
        if mean_tardiness < self.last_tardiness:
            r2 = 1
        elif mean_tardiness == self.last_tardiness:
            r2 = 0
        else:
            r2 = -1
        self.last_tardiness = mean_tardiness

        r3 = 0
        return [r1, r2, r3]

    def ra_reward(self, mach_index, op):
        estimate_use_ratio = np.zeros(self.mch_num)
        estimate_ava_t = np.zeros(self.mch_num)
        estimate_tardiness = np.zeros(len(self.jobs))

        for i in range(self.mch_num):
            m = self.machines[i]
            estimate_use_ratio[i] = m.get_estimate_use_ratio()
            estimate_ava_t[i] = m.get_estimate_ava_time()

        est_cur = estimate_ava_t.mean()

        for j in range(len(self.jobs)):
            if self.finished_flag[j] is True:
                continue
            estimate_tardiness[j] = self.jobs[j].get_tardiness(est_cur, True)

        mean_est_ur = estimate_use_ratio.mean()
        r1 = 0

        self.last_est_ur = mean_est_ur

        mean_est_tardiness = estimate_tardiness.mean()
        if mean_est_tardiness < self.last_est_tardiness:
            r2 = 1
        elif mean_est_tardiness == self.last_est_tardiness:
            r2 = 0
        else:
            r2 = -1

        self.last_est_tardiness = mean_est_tardiness

        self.sum_ect += op.get_pt(mach_index)
        self.sum_pt += op.get_ect(mach_index)
        ep_ratio = self.sum_ect / self.sum_pt
        if ep_ratio > self.last_ep_ratio:
            r3 = 1
        elif ep_ratio == self.last_ep_ratio:
            r3 = 0
        else:
            r3 = -1
        self.last_ep_ratio = ep_ratio

        return [r1, r2, r3]

    def sa_step(self, mach_index, sa_action, t):
        job_index = self.sequence_rule(mach_index, sa_action, t)
        # 从buffer选择job_index的工件进行加工
        op, start, end = self.machines[mach_index].sa_step(job_index)
        op.sa_step(start, end)
        self.jobs[job_index].sa_step(start, end)

        if self.jobs[job_index].pre_no == self.jobs[job_index].ops_num:
            self.finished_flag[job_index] = True
        if self.sa_done(end):
            done = True
        else:
            done = False
        reward = self.sa_reward(mach_index)
        return job_index, reward, done, end

    def ra_step(self, job_index, ra_action, t):
        if len(self.jobs[job_index].pre_op.ava_mach_number) == 1:
            mach_index = self.jobs[job_index].pre_op.ava_mach_number[0]
        else:
            mach_index = self.route_rule(job_index, ra_action)
        # 将job_index的pre_op插入mach_index的buffer中
        op = self.jobs[job_index].pre_op
        start = self.jobs[job_index].pre_start

        pt, ect = self.machines[mach_index].ra_step(op, start)
        self.jobs[job_index].ra_step(ect)
        reward = self.ra_reward(mach_index, op)
        op.ra_step(pt, ect, mach_index)

        if self.jobs[job_index].ops_dispatch_no == self.jobs[job_index].ops_num:
            self.routed_flag[job_index] = True
        done = self.ra_done(t)
        return reward, done

    def step(self, ra, t):
        if not self.sa_done(t):
            mach_index, t = self.cal_schedule_time(ra)
            sa_state = self.get_sa_state(mach_index)
        else:
            sa_state = np.zeros(self.sa_state_dim) + 1e-4
            mach_index = None
            t = None
        self._discount_update()
        return sa_state, mach_index, t

    def _discount_update(self, gamma=5e-4, beta=7e-4):
        self.discount1 += gamma
        if self.d_flag:
            self.discount2 += 6e-5
        else:
            self.discount2 += beta
        if self.discount1 >= 1:
            self.discount1 = 0.99
        if self.discount2 >= 1:
            self.d_flag = True
            self.discount2 = 0.99

    def sequence_rule(self, mach_index, action, t):
        if mach_index is None:
            raise Exception('mach_index is NoneType')
        job_inds = self.machines[mach_index].get_sa_job_index()  # 得到buffer中最近可用的工件编号
        index = job_dispatch(action, self.jobs[job_inds], t)
        return job_inds[index]

    def route_rule(self, job_index, action):
        if self.jobs[job_index].pre_no == self.jobs[job_index].ops_num:
            raise Exception('job {} pre_no is equal ops_num, can not be routed again'.format(job_index + 1))
        op = self.jobs[job_index].pre_op
        if len(op.ava_mach_number) == 1:
            return op.ava_mach_number[0]
        else:
            index = machine_dispatch(self.machines[op.ava_mach_number], op, action)
            return op.ava_mach_number[index]

    def get_first_ava_mach(self, mach_inds):
        ava_t = self.machines[mach_inds[0]].get_estimate_ava_time()
        index = mach_inds[0]
        for mach_index in mach_inds[1:]:
            new_ava_t = self.machines[mach_index].get_estimate_ava_time()
            if new_ava_t < ava_t:
                ava_t = new_ava_t
                index = mach_index
        return index

    def reset(self, ra=None, t=0):
        job_index = np.arange(self.job_num)
        # np.random.shuffle(job_index) # 保证稳定性取消初始index顺序打乱
        ra_state = self.get_ra_state(job_index[0])

        for i in range(0, self.job_num):
            j = job_index[i]
            action = ra.choose_action(ra_state)
            reward, done = self.ra_step(j, action, t)
            if i != self.job_num - 1:
                ra_state_ = self.get_ra_state(job_index[i + 1])
            else:
                ra_state_ = self.get_ra_state(job_index[i])
            ra.store(ra_state, action, reward, done)
            ra_state = ra_state_
        mach_index, t = self.cal_schedule_time(ra)
        sa_state = self.get_sa_state(mach_index)
        return sa_state, mach_index, t

    def check_njob_arrival(self, t):
        if self.new_job_cnt != len(self.new_jobs):
            if self.new_arrival_t[self.new_job_cnt] <= t:
                return True
        else:
            return False

    def njob_insert(self):
        njob = self.new_jobs[self.new_job_cnt]
        self.jobs = np.append(self.jobs, njob)
        job_index = len(self.jobs) - 1
        self.jobs[-1].job_index = job_index
        for i in range(self.jobs[-1].ops_num):
            op = self.jobs[-1].ops[i]
            op.job_index = job_index
        self.routed_flag = np.append(self.routed_flag, False)
        self.finished_flag = np.append(self.finished_flag, False)
        if self.new_job_cnt != self.new_num:
            self.new_job_cnt += 1
        return job_index

    def njob_ra_state(self, job_index):
        t = self.jobs[job_index].arrival_t
        ra_state = self.get_ra_state(job_index)
        return ra_state

    def njob_route(self, job_index, t, ra=None):
        ra_state = self.njob_ra_state(job_index)
        action = ra.choose_action(ra_state)
        mach_index = self.route_rule(job_index, action)
        reward, done = self.ra_step(job_index, action, t)
        ra.store(ra_state, action, reward, done)
        return mach_index

    def cal_objective(self):
        use_ratio = []
        ect = 0
        tardiness = 0
        for i in range(self.mch_num):
            m = self.machines[i]
            use_ratio.append(m.cal_use_ratio())
            ect += m.cal_total_ect()
        for j in range(len(self.jobs)):
            tardiness += self.jobs[j].get_tardiness(self.t)
        mean_use_ratio = np.array(use_ratio).mean()
        return [mean_use_ratio, ect, tardiness]

    def render(self, t=0.5):
        plt.title('Gantt chart of scheduling')
        plt.yticks(np.arange(len(self.machines) + 1))
        for i in range(len(self.machines)):
            m = self.machines[i]
            for j in range(len(m.finished_op)):
                op = m.finished_op[j]
                job_index, op_index = op.job_index, op.op_index
                c_index = job_index % len(self.color)
                start, end = op.start, op.end
                pt = end - start
                plt.barh(i + 0.5, pt, height=1, left=start, align='center', color=self.color[c_index], edgecolor='grey')
                # plt.text(start + pt / 8, i, 'J{}-OP{}\n{}'.machine_format(job_index + 1, op_index + 1, pt), fontsize=10,color='tan')
            for j in range(m.break_cnt):
                start = m.break_t[j]
                rep_t = m.rep_t[j]
                plt.barh(i + 0.5, rep_t, height=1, left=start, align='center', color='black', edgecolor='grey')
                # plt.text(start + rep_t / 8, i, 'B{}-R{}'.machine_format(j + 1, rep_t), fontsize=10, color='tan')
        plt.xlabel('time')
        plt.ylabel('machine')
        plt.pause(t)
        plt.close()
