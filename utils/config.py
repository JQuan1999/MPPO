import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15, help='every weight learn epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='learning batch size')
    parser.add_argument('--a_update_step', type=int, default=10, help='actor learning step')
    parser.add_argument('--c_update_step', type=int, default=10, help='critic learning step')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount1 reward')
    parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
    parser.add_argument('--weight_size', type=int, default=100, help='sample weight')
    parser.add_argument('--objective', type=int, default=3, help='objective size')
    parser.add_argument('--sa_state_dim', type=int, default=12, help='sequence agent state dim')
    parser.add_argument('--ra_state_dim', type=int, default=12, help='route agent state dim')
    parser.add_argument('--sa_action_space', type=int, default=4, help='sequence agent action space')
    parser.add_argument('--ra_action_space', type=int, default=4, help='route agent action space')

    parser.add_argument('--agent_eval_savedir', type=str, default='./log/eval/agent', help='eval result saved path')
    parser.add_argument('--rule_eval_savedir', type=str, default='./log/eval/rule', help='rule combination eval result saved path')
    parser.add_argument('--randrule_eval_savedir', type=str, default='./log/eval/randrule', help='rand rule combination eval result saved path')
    parser.add_argument('--compared_eval_result', type=str, default='./log/eval/compared',
                        help='the eval result of all compared methods and to compute the pareto metric result')
    parser.add_argument('--metric_save_dir', type=str, default='./log/pareto', help='the save path of pareto metric result')
    parser.add_argument('--metric_file', type=str, default='metric.json', help='the metric result filename')
    parser.add_argument('--param_comb_ckpt', type=str, default='./param/param_experiment',
                        help='the path to save the sa and ra agent ckpt of different params combination')

    parser.add_argument('--train_data', type=str, default='./data/train/j30_m20_n40/')
    parser.add_argument('--test_data', type=str, default='./data/test')
    parser.add_argument('--sa_ckpt_path', type=str, default='./param/pareto_weight/sa', help='sa ckpt dirname')
    parser.add_argument('--ra_ckpt_path', type=str, default='./param/pareto_weight/ra', help='ra ckpt dirname')
    parser.add_argument('--weight_path', type=str, default='./param/pareto_weight/weight.npy', help='weight path')
    args = parser.parse_args()
    return args