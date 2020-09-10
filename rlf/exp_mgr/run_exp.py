from rlf.exp_mgr import config_mgr
import os.path as osp
import os
import argparse
import libtmux
import sys
import random
import string


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sess-id', type=int, default=-1,
                        help='tmux session id to connect to. If unspec will run in current window')
    parser.add_argument('--sess-name', default=None, type=str,
                        help='tmux session name to connect to')
    parser.add_argument('--cmd', type=str, required=True,
                        help='list of commands to run')
    parser.add_argument('--seed', type=str, default=None)

    parser.add_argument('--cd', default='1', type=str,
                        help='String of CUDA_VISIBLE_DEVICES=(example: \"1 2\")')
    parser.add_argument('--cfg', type=str, default='./config.yaml')
    return parser


def add_on_args(spec_args):
    spec_args = ['"' + x + '"' if ' ' in x else x for x in spec_args]
    return ((' '.join(spec_args)))


def get_cmds(cmd_path, spec_args):
    try:
        open_cmd = osp.join(cmd_path + '.cmd')
        print('opening', open_cmd)
        with open(open_cmd) as f:
            cmds = f.readlines()
    except:
        raise ValueError(f"Command at {cmd_path} does not exist")

    # pay attention to comments
    cmds = list(filter(lambda x: not (x.startswith('#') or x == '\n'), cmds))
    cmds = [cmd.rstrip() + " " for cmd in cmds]

    # Check if any commands are references to other commands
    all_ref_cmds = []
    for i, cmd in enumerate(cmds):
        if cmd.startswith('R:'):
            cmd_parts = cmd.split(':')[1].split(' ')
            ref_cmd_loc = cmd_parts[0]
            full_ref_cmd_loc = osp.join(config_mgr.get_prop('cmds_loc'),
                    ref_cmd_loc)
            ref_cmds = get_cmds(full_ref_cmd_loc.rstrip(), [*cmd_parts[1:], *spec_args])
            all_ref_cmds.extend(ref_cmds)
    cmds = list(filter(lambda x: not x.startswith('R:'), cmds))

    cmds = [cmd + add_on_args(spec_args) for cmd in cmds]

    cmds.extend(all_ref_cmds)
    return cmds


def get_tmux_window(sess_name, sess_id):
    server = libtmux.Server()

    if sess_name is None:
        tmp = server.list_sessions()
        sess = server.get_by_id('$%i' % sess_id)
    else:
        sess = server.find_where({"session_name": sess_name})
    if sess is None:
        raise ValueError('invalid session id')

    return sess.new_window(attach=False, window_name="auto_proc")


def execute_command_file(cmd_path, add_args_str, cd, sess_name, sess_id, seed):
    cmds = get_cmds(cmd_path, add_args_str)

    if seed is not None and len(seed.split(',')) > 1:
        seeds = seed.split(',')
        common_id = ''.join(random.sample(string.ascii_uppercase + string.digits, k=2))

        def transform_prefix(s, common_id):
            prefix_parts = s.split('--prefix ')
            before_prefix = '--prefix '.join(prefix_parts[:-1])
            prefix = prefix_parts[-1]
            parts = prefix.split(' ')
            prefix = parts[0]
            after_prefix = ' '.join(parts[1:])
            return before_prefix + f"--prefix {prefix}-{common_id} " + after_prefix
        cmds = [transform_prefix(cmd, common_id) for cmd in cmds]
        cmds = [cmd + f" --seed {seed}"  for cmd in cmds for seed in seeds]
        rnd_id = ''.join(random.sample(string.ascii_uppercase + string.digits, k=2))
    elif seed is not None:
        cmds = [x + f" --seed {seed}" for x in cmds]
    add_on = ''

    if sess_id == -1:
        if len(cmds) == 1:
            exec_cmd = cmds[0]
            if cd != '-1':
                exec_cmd = 'CUDA_VISIBLE_DEVICES=' + cd + ' ' + exec_cmd + ' ' + add_on
            else:
                exec_cmd = exec_cmd + ' ' + add_on
            print('executing ', exec_cmd)
            os.system(exec_cmd)
        else:
            raise ValueError('Running multiple jobs. You must specify tmux session id')
    else:

        exp_files = []
        for cmd in cmds:
            new_window = get_tmux_window(sess_name, sess_id)
            cmd += ' ' + add_on
            print('running full command %s\n' % cmd)

            # Send the keys to run the command
            last_pane = new_window.attached_pane
            last_pane.send_keys(cmd, enter=False)
            pane = new_window.split_window(attach=False)
            pane.set_height(height=50)
            pane.send_keys('source deactivate')

            conda_env = config_mgr.get_prop('conda_env')
            pane.send_keys('source activate ' + conda_env)
            pane.enter()
            if cd != '-1':
                pane.send_keys('export CUDA_VISIBLE_DEVICES=' + cd)
                pane.enter()
            pane.send_keys(cmd)
            pane.enter()

        print('everything should be running...')

def full_execute_command_file():
    parser = get_arg_parser()
    print(os.getcwd())
    args, rest = parser.parse_known_args()
    config_mgr.init(args.cfg)

    cmd_path = osp.join(config_mgr.get_prop('cmds_loc'), args.cmd)
    execute_command_file(cmd_path, rest, args.cd, args.sess_name, args.sess_id,
            args.seed)

if __name__ == '__main__':
    full_execute_command_file()

