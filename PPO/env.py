# author by 蒋权
import numpy as np
import sys
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
    def __init__(self, instance, t=5000):
        self.jobs, self.machines, self.new_jobs = get_data(instance)
        self.new_arrival_t = [self.new_jobs[i].arrival_t for i in range(len(self.new_jobs))]
        self.job_num = len(self.jobs)
        self.mch_num = len(self.machines)
        self.new_num = len(self.new_jobs)
        self.sa_state_dim = 15
        self.ra_state_dim = 13
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
        self.w = None
        self.color = np.array(color).reshape(-1, 3).tolist()
        self.routed_flag = np.zeros(len(self.jobs)).astype(bool)
        self.finished_flag = np.zeros(len(self.jobs)).astype(bool)

    def cal_reward(self, job_index, machine_index, objective):
        if objective == 0:  # 目标函数1 完工时间
            # t+1阶段利用率
            use_ratio_1 = self.machines[machine_index].cal_use_ratio()
            if len(self.machines[machine_index].finished_op) != 1:
                use_ratio_2 = self.machines[machine_index].cal_use_ratio(
                    len(self.machines[machine_index].finished_op) - 1)  # t阶段利用率
            else:
                use_ratio_2 = 0
            if use_ratio_1 > use_ratio_2:
                reward = 1
            elif use_ratio_1 > use_ratio_2 * 0.95:
                reward = 0
            else:
                reward = -1
            return reward
        elif objective == 1:  # 目标函数2 延迟时间
            tcur = np.array([m.ava_t for m in self.machines]).mean()  # 机器平均可用时间
            weight_tardiness = np.zeros(len(self.jobs))
            for i in range(len(self.jobs)):
                weight_tardiness[i] = self.jobs[i].get_tardiness(tcur)
            mean_tardiness = weight_tardiness.mean()
            if self.last_tardiness is None:
                return 1
            if mean_tardiness < self.last_tardiness:
                reward = 1
            elif mean_tardiness * 0.9 < self.last_tardiness:
                reward = 0
            else:
                reward = -1
            self.last_tardiness = mean_tardiness
            return reward
        elif objective == 2:  # 目标函数3 累计能耗
            index = len(self.machines[machine_index].finished_op)
            ect_1 = np.array(self.machines[machine_index].ects[:index]).mean()
            ect_2 = np.array(self.machines[machine_index].ects[:index - 1]).mean()
            if ect_1 < ect_2:
                return 1
            elif ect_1 == ect_2:
                return 0
            else:
                return -1

    # def get_oa_state(self, t):
    #     mach_use_ratio = np.zeros(self.mch_num)  # 机器利用率
    #     mach_ect = np.zeros(self.mch_num)  # 机器能耗
    #     job_tardiness = np.zeros(self.job_num)  # 工件延期时间
    #     for m in range(self.mch_num):
    #         mach_use_ratio[m] = self.machines[m].get_use_ratio()
    #         mach_ect[m] = self.machines[m].get_ect()
    #     for j in range(self.jobs):
    #         job_tardiness[j] = self.jobs[j].get_tardiness(t)
    #     mean_use_ratio = mach_use_ratio.mean()
    #     min_use_ratio = mach_use_ratio.mean()
    #     use_ratio = np.array([mean_use_ratio, min_use_ratio])
    #     use_ratio = (use_ratio / use_ratio.max()).tolist()
    #
    #     sum_mach_ect = mach_ect.sum()
    #     mean_mach_ect = mach_ect.mean()
    #     mach_ect = np.array([sum_mach_ect, mean_mach_ect])
    #     mach_ect = (mach_ect / mach_ect.max()).tolist()
    #
    #     mean_job_tardiness = job_tardiness.mean()
    #     min_job_tardiness = job_tardiness.min()
    #     job_tardiness = np.array([mean_job_tardiness, min_job_tardiness])
    #     job_tardiness = (job_tardiness / job_tardiness.max()).tolist()
    #
    #     oa_state = use_ratio + mach_ect + job_tardiness
    #     return oa_state

    def get_sa_state(self, mach_index, t):
        if len(self.machines[mach_index].buffer_op) == 0:
            return np.zeros(self.sa_state_dim) + 1e-4
        ava_t, mean_start, min_start, mean_pt, min_pt, job_inds = self.machines[mach_index].get_state()  # 加工机器特征
        slack = np.zeros(len(job_inds))
        remain = np.zeros(len(job_inds))
        finish_rate = np.zeros(len(job_inds))
        for j in range(len(job_inds)):
            if self.finished_flag[j] is True:
                continue
            job_index = job_inds[j]
            slack[j] = self.jobs[job_index].get_slack_time(t)  # slack为负值表示已延期
            remain[j] = self.jobs[job_index].get_remain_pt()
            finish_rate = self.jobs[job_index].get_finish_rate()
        slack_mean = slack.mean()
        slack_min = slack.min()
        remain_mean = remain.mean()
        remain_min = remain.min()
        fr_mean = finish_rate.mean()
        fr_min = finish_rate.min()
        sys_jobs_num = len(self.finished_flag) - self.finished_flag.sum()
        op_rate = len(self.machines[mach_index].buffer_op) / sys_jobs_num

        norm1 = np.array([ava_t, mean_start, min_start])  # size 3
        if norm1.max() == 0:
            norm1 = norm1.tolist()
        else:
            norm1 = (norm1 / norm1.max()).tolist()

        norm2 = np.array([mean_pt, min_pt, slack_mean, slack_min, remain_mean, remain_min])  # size 6
        if norm2.max() == 0:
            norm2 = norm2.tolist()
        else:
            norm2 = (norm2 / norm2.max()).tolist()
        sa_state = norm1 + norm2 + [fr_mean, fr_min, op_rate] + self.w  # size 15
        return sa_state

    def get_ra_state(self, job_index, t):
        ava_t = np.zeros(self.mch_num)  # 机器可用时间
        est_ava_t = np.zeros(self.mch_num)  # 机器预估完成buffer工件时间
        for i in range(self.mch_num):
            m = self.machines[i]
            ava_t[i] = m.ava_t
            est_ava_t[i] = m.get_estimate_ava_time()
        ava_t_mean = ava_t.mean()
        ava_t_min = ava_t.min()
        est_ava_t_mean = est_ava_t.mean()
        est_ava_t_min = est_ava_t.min()

        start = self.jobs[job_index].pre_start  # 工件当前工序开始加工时间
        finish_rate = self.jobs[job_index].get_finish_rate()  # 工件完成率
        slack = self.jobs[job_index].get_slack_time(t)
        remain = self.jobs[job_index].get_remain_pt()
        ep_rate = self.jobs[job_index].pre_op.get_ep_rate()  # 能耗与加工时间之比
        epr_mean = ep_rate.mean()
        epr_min = ep_rate.min()

        norm1 = np.array([ava_t_mean, ava_t_min, est_ava_t_mean, est_ava_t_min, start])  # size 5
        if norm1.max() == 0:
            norm1 = norm1.tolist()
        else:
            norm1 = (norm1 / norm1.max()).tolist()

        norm2 = np.array([slack, remain])  # size 2
        if norm2.max() == 0:
            norm2 = norm2.tolist()
        else:
            norm2 = (norm2 / norm2.max()).tolist()

        ra_state = norm1 + norm2 + [finish_rate, epr_mean, epr_min] + self.w  # size 13
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
            choose = self.njob_route(job_index, ra, t)
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

    def sa_reward(self, ep_r):
        mach_use_rats = np.zeros(len(self.machines))
        ava_t = np.zeros(len(self.machines))
        for i in range(len(self.machines)):
            mach_use_rats[i] = self.machines[i].cal_use_ratio()
            ava_t[i] = self.machines[i].ava_t
        mean_ur = mach_use_rats.mean()
        r1 = mean_ur - self.last_use_ratio
        self.last_use_ratio = mean_ur

        tcur = ava_t.mean()
        tardiness = np.zeros(len(self.jobs))
        for j in range(len(self.jobs)):
            if self.finished_flag[j] is True:
                continue
            tardiness[j] = self.jobs[j].get_tardiness(tcur)
        mean_tardiness = tardiness.mean()
        r2 = self.last_tardiness - mean_tardiness
        self.last_tardiness = mean_tardiness

        r3 = ep_r
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
        r1 = mean_est_ur - self.last_est_ur
        self.last_est_ur = mean_est_ur

        mean_est_tardiness = estimate_tardiness.mean()
        r2 = self.last_est_tardiness - mean_est_tardiness
        self.last_est_tardiness = mean_est_tardiness

        self.sum_ect += op.get_pt(mach_index)
        self.sum_pt += op.get_ect(mach_index)
        ep_ratio = self.sum_ect / self.sum_pt
        r3 = ep_ratio - self.last_ep_ratio
        self.last_ep_ratio = ep_ratio

        return [r1, r2, r3]

    def sa_step(self, mach_index, job_index):
        if self.jobs[job_index].pre_op.ep_r is None:
            raise Exception('ep_r of op is None')
        ep_r = self.jobs[job_index].pre_op.ep_r

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
        reward = self.sa_reward(ep_r)
        return reward, done, end

    def ra_step(self, mach_index, job_index, t):
        # 将job_index的pre_op插入mach_index的buffer中
        op = self.jobs[job_index].pre_op
        start = self.jobs[job_index].pre_start

        pt, ect = self.machines[mach_index].ra_step(op, start)
        self.jobs[job_index].ra_step(ect)
        reward = self.ra_reward(mach_index, op)
        op.ra_step(pt, ect, mach_index, reward[-1])

        if self.jobs[job_index].ops_dispatch_no == self.jobs[job_index].ops_num:
            self.routed_flag[job_index] = True
        done = self.ra_done(t)
        return reward, done

    def step(self, ra, t):
        # reward = self.cal_reward(job_index, mach_index, objective)
        if not self.sa_done(t):
            mach_index, t = self.cal_schedule_time(ra)
            sa_state = self.get_sa_state(mach_index, t)
        else:
            sa_state = np.zeros(self.sa_state_dim) + 1e-4
            mach_index = None
            t = None
        return sa_state, mach_index, t

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

    def reset(self, ra, t=0):
        job_index = np.arange(self.job_num)
        np.random.shuffle(job_index)

        ra_state = self.get_ra_state(job_index[0], t)
        for i in range(0, self.job_num):
            j = job_index[i]
            action = ra.choose_action(ra_state)
            if len(self.jobs[j].pre_op.ava_mach_number) == 1:
                mach_index = self.jobs[j].pre_op.ava_mach_number[0]
            else:
                mach_index = self.route_rule(j, action)
            reward, done = self.ra_step(mach_index, j, t)
            if i != self.job_num - 1:
                ra_state_ = self.get_ra_state(job_index[i + 1], t)
            else:
                ra_state_ = self.get_ra_state(job_index[i], t)
            ra.store(ra_state, action, reward, done)
            ra_state = ra_state_
        mach_index, t = self.cal_schedule_time(ra)
        sa_state = self.get_sa_state(mach_index, t)
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
        ra_state = self.get_ra_state(job_index, t)
        return ra_state

    def njob_route(self, job_index, ra, t):
        ra_state = self.njob_ra_state(job_index)
        action = ra.choose_action(ra_state)
        mach_index = self.route_rule(job_index, action)
        reward, done = self.ra_step(mach_index, job_index, t)
        ra.store(ra_state, action, reward, done)
        return mach_index

    def cal_objective(self):
        cmax = 0
        ect = 0
        tardiness = 0
        for i in range(self.mch_num):
            m = self.machines[i]
            cmax = max(cmax, m.ava_t)
            ect += m.get_total_ect()
        for j in range(len(self.jobs)):
            tardiness += self.jobs[j].get_ftardiness(self.t)
        return [cmax, ect, tardiness]

    def render(self, t=1):
        plt.title('gantt chart of scheduling')
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
                plt.text(start + pt / 8, i, 'J{}-OP{}\n{}'.format(job_index + 1, op_index + 1, pt), fontsize=10,
                         color='tan')
            for j in range(m.break_cnt):
                start = m.break_t[j]
                rep_t = m.rep_t[j]
                plt.barh(i + 0.5, rep_t, height=1, left=start, align='center', color='yellow', edgecolor='grey')
                plt.text(start + rep_t / 8, i, 'B{}-R{}'.format(j + 1, rep_t), fontsize=10, color='tan')
        plt.xlabel('time')
        plt.ylabel('machine')
        plt.pause(t)
        plt.close()
