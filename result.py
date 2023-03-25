# author by 蒋权
import json

import numpy as np
import matplotlib.pyplot as plt


def show_objective1(obj_values, rule, data, obj):
    labels = np.array(['Nk10_M10_Ns20', 'Nk10_M10_Ns50', 'Nk10_M10_Ns100',
                       'Nk20_M20_Ns20', 'Nk20_M20_Ns50', 'Nk20_M20_Ns100',
                       'Nk30_M30_Ns20', 'Nk30_M30_Ns50', 'Nk30_M30_Ns100'])
    plt.figure()
    # xtick = ['Agent'] + rule + ['Ran\ndom']
    xtick = ['Agent'] + rule
    if obj == 0:
        plt.title('USE RATIO')
    if obj == 1:
        plt.title('TOTAL ECOST')
    if obj == 2:
        plt.title('TOTAL TARDINESS')
    for i in range(len(xtick)):
        xtick[i] = '\n'.join(xtick[i].split('_'))
    xline_space = np.linspace(0, len(xtick), num=len(xtick))
    plt.xticks(xline_space, xtick, fontsize=6)
    plt.xlabel('Rule')
    plt.ylabel('Rank')
    plt.yticks(range(len(xtick) + 5))
    plt.ylim(0, len(rule) + 8)
    color = ['lightgray', 'lightcoral', 'tomato', 'sienna', 'darkorange', 'wheat', 'lightsteelblue', 'slategray',
             'olive']
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', '*', '+']
    for i in range(obj_values.shape[0]):
        if obj == 0:
            index = np.argsort(obj_values[i])
        else:
            index = np.argsort(-obj_values[i])
        rank = np.arange(1, obj_values.shape[1] + 1)[::-1]
        sort_rank = np.zeros(obj_values.shape[1])
        sort_rank[index] = rank
        plt.scatter(xline_space, sort_rank, marker=mark[i], color=color[i], label=labels[i])
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    plt.show()


def show_objective2(obj_values, rule, data, obj):
    labels = np.array(['Nk10_M10_Ns20', 'Nk10_M10_Ns50', 'Nk10_M10_Ns100',
                       'Nk20_M20_Ns20', 'Nk20_M20_Ns50', 'Nk20_M20_Ns100',
                       'Nk30_M30_Ns20', 'Nk30_M30_Ns50', 'Nk30_M30_Ns100'])
    plt.figure()
    # xtick = ['Agent'] + rule + ['Ran\ndom']
    xtick = ['Agent'] + rule
    for i in range(len(xtick)):
        xtick[i] = '\n'.join(xtick[i].split('_'))
    if obj == 0:
        plt.ylim(0, 1)
        plt.title('USE RATIO')
    if obj == 1:
        plt.title('TOTAL ECOST')
        plt.ticklabel_format(axis='y', style='sci', scilimits=[-1, 2])
        plt.ylim(0, obj_values.max() * 1.21)
    if obj == 2:
        plt.title('TOTAL TARDINESS')
        plt.ticklabel_format(axis='y', style='sci', scilimits=[-1, 2])
        plt.ylim(0, obj_values.max() * 1.21)
    xline_space = np.linspace(0, len(xtick), num=len(xtick))
    plt.xticks(xline_space, xtick, fontsize=6)
    plt.xlabel('Rule')
    plt.ylabel('Value')
    color = ['lightgray', 'lightcoral', 'tomato', 'sienna', 'darkorange', 'wheat', 'lightsteelblue', 'slategray',
             'olive']
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', '*', '+']
    for i in range(obj_values.shape[0]):
        plt.plot(xline_space, obj_values[i, :], marker=mark[i], color=color[i], label=labels[i])
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    plt.show()


def show_objective(obj_values, rule, obj):
    labels = np.array(['Nk10_M10_Ns20', 'Nk10_M10_Ns50', 'Nk10_M10_Ns100',
                       'Nk20_M20_Ns20', 'Nk20_M20_Ns50', 'Nk20_M20_Ns100',
                       'Nk30_M30_Ns20', 'Nk30_M30_Ns50', 'Nk30_M30_Ns100'])
    color = ['lightgray', 'lightcoral', 'tomato', 'sienna', 'darkorange', 'wheat', 'lightsteelblue', 'slategray',
             'olive']
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', '*', '+']
    # rank值
    _, axes = plt.subplots(1, 2, figsize=(13, 8))
    xlabels = ['Agent'] + rule
    for i in range(len(xlabels)):
        xlabels[i] = '\n'.join(xlabels[i].split('_'))
    if obj == 0:
        axes[0].set_title('Use ratio')
        axes[1].set_title('Use ratio')
        axes[1].set_ylim(0, 1)
    elif obj == 1:
        axes[0].set_title('Energy cost')
        axes[1].set_title('Energy cost')
        axes[1].ticklabel_format(axis='y', style='sci', scilimits=[-1, 2])
        axes[1].set_ylim(0, obj_values.max() * 1.21)
    elif obj == 2:
        axes[0].set_title('Tardiness')
        axes[1].set_title('Tardiness')
        axes[1].ticklabel_format(axis='y', style='sci', scilimits=[-1, 2])
        axes[1].set_ylim(0, obj_values.max() * 1.21)
    xline_space = np.linspace(0, len(xlabels), num=len(xlabels))
    # axes0
    axes[0].set_xticks(xline_space)
    axes[0].set_xticklabels(xlabels, fontsize=6)
    axes[0].set_xlabel('Rule')
    axes[0].set_ylabel('Rank')
    axes[0].set_yticks(range(len(xlabels) + 5))
    axes[0].set_ylim(0, len(rule) + 5)
    for i in range(obj_values.shape[0]):
        if obj == 0:
            index = np.argsort(obj_values[i])
        else:
            index = np.argsort(-obj_values[i])
        rank = np.arange(1, obj_values.shape[1] + 1)[::-1]
        sort_rank = np.zeros(obj_values.shape[1])
        sort_rank[index] = rank
        axes[0].scatter(xline_space, sort_rank, marker=mark[i], color=color[i], label=labels[i])
    axes[0].legend(loc='upper right', fontsize=6, ncol=3)

    # axes1
    axes[1].set_xticks(xline_space)
    axes[1].set_xticklabels(xlabels, fontsize=6)
    axes[1].set_xlabel('Rule')
    axes[1].set_ylabel('Value')
    for i in range(obj_values.shape[0]):
        axes[1].plot(xline_space, obj_values[i, :], marker=mark[i], color=color[i], label=labels[i])
    axes[1].legend(loc='upper right', fontsize=6, ncol=3)

    plt.show()


def show_result(agent_log, rule_log, randm_log, test_data):
    with open(agent_log, 'r') as f:
        alog = json.load(f)
    with open(rule_log, 'r') as f:
        rlog = json.load(f)
    with open(randm_log, 'r') as f:
        rmlog = json.load(f)
    rule = np.array([key for key in rlog]).tolist()
    # rule = np.random.choice(rule, size=20, replace=False).tolist()
    for o in range(3):
        # obj_values = np.zeros((len(test_data), len(rule)+2))
        obj_values = np.zeros((len(test_data), len(rule) + 1))
        for i in range(test_data.shape[0]):
            obj_values[i][0] = alog[test_data[i]][o]
            for r in range(len(rule)):
                obj_values[i][r + 1] = rlog[rule[r]][test_data[i]][o]
            # obj_values[i][-1] = rmlog[test_data[i]][o]
        show_objective(obj_values, rule, o)
    print('end')


if __name__ == '__main__':
    agent_log = './log/eval/09-04-13-53-agent.json'
    rule_log = './log/eval/09-04-14-19-rule.json'
    randm_log = './log/eval/09-04-13-24-randRule.json'
    test_data = np.array(['j10_m10_n20', 'j10_m10_n50', 'j10_m10_n100',
                          'j20_m20_n20', 'j20_m20_n50', 'j20_m20_n100',
                          'j30_m30_n20', 'j30_m30_n50', 'j30_m30_n100'])
    show_result(agent_log, rule_log, randm_log, test_data)