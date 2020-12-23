from rlf.exp_mgr import config_mgr
import os.path as osp
import os
import argparse
import libtmux
import sys
import random
import string
import uuid


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sess-id', type=int, default=-1,
                        help='tmux session id to connect to. If unspec will run in current window')
    parser.add_argument('--sess-name', default=None, type=str,
                        help='tmux session name to connect to')
    parser.add_argument('--cmd', type=str, required=True,
                        help='list of commands to run')
    parser.add_argument('--seed', type=str, default=None)
    parser.add_argument('--st', type=str, default=None, help="Slum type [long, short]")
    parser.add_argument('--c', type=int, default=6, help="""
            Number of cpus for SLURM job
            """)
    parser.add_argument('--g', type=int, default=1, help="""
            Number of gpus for SLURM job
            """)
    parser.add_argument('--nodes', type=int, default=1, help="""
            Number of nodes for SLURM job
            """)
    parser.add_argument('--cd', default='1', type=str,
                        help='String of CUDA_VISIBLE_DEVICES=(example: \"1 2\")')
    parser.add_argument('--cfg', type=str, default='./config.yaml')
    parser.add_argument('--pt-proc', type=int, default=-1)
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

def transform_prefix(s, common_id):
    if '--prefix' in s:
        use_split = '--prefix '
    else:
        use_split = 'PREFIX '

    prefix_parts = s.split(use_split)
    before_prefix = use_split.join(prefix_parts[:-1])
    prefix = prefix_parts[-1]
    parts = prefix.split(' ')
    prefix = parts[0]
    after_prefix = ' '.join(parts[1:])
    return before_prefix + f"{use_split} {prefix}-{common_id} " + after_prefix

def execute_command_file(cmd_path, add_args_str, cd, sess_name, sess_id, seed,
        args):
    cmds = get_cmds(cmd_path, add_args_str)

    if seed is not None and len(seed.split(',')) > 1:
        seeds = seed.split(',')
        common_id = ''.join(random.sample(string.ascii_uppercase + string.digits, k=2))

        cmds = [transform_prefix(cmd, common_id) for cmd in cmds]
        cmds = [cmd + f" --seed {seed}"  for cmd in cmds for seed in seeds]
    elif seed is not None:
        cmds = [x + f" --seed {seed}" for x in cmds]
    add_on = ''

    if len(cmds) > 0:
        # Make sure all the commands share the last part of the prefix so they can
        # find each other. The name is long because its really bad if a job
        # finds the wrong other job.
        common_id = ''.join(random.sample(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6))
        cmds = [transform_prefix(cmd, common_id) for cmd in cmds]

    if args.pt_proc != -1:
        pt_dist_str = f"python -u -m torch.distributed.launch --use_env --nproc_per_node {args.pt_proc} "
        def make_dist_cmd(x):
            parts = x.split(' ')
            runf = None
            for i, part in enumerate(parts):
                if '.py' in part:
                    runf = i
                    break

            if runf is None:
                raise ValueError('Could not split command')

            rest = ' '.join(parts[runf:])
            return pt_dist_str + rest

        cmds = [make_dist_cmd(x) for x in cmds]

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
        cd = cd.split(',')
        if len(cd) == 1:
            cd = [cd[0] for _ in cmds]
        for i, cmd in enumerate(cmds):
            new_window = get_tmux_window(sess_name, sess_id)
            cmd += ' ' + add_on
            print('running full command %s\n' % cmd)

            # Send the keys to run the command
            conda_env = config_mgr.get_prop('conda_env')
            if args.st is None:
                last_pane = new_window.attached_pane
                last_pane.send_keys(cmd, enter=False)
                pane = new_window.split_window(attach=False)
                pane.set_height(height=50)
                pane.send_keys('source deactivate')

                pane.send_keys('source activate ' + conda_env)
                pane.enter()
                if cd[i] != '-1':
                    pane.send_keys('export CUDA_VISIBLE_DEVICES=' + cd[i])
                    pane.enter()
                pane.send_keys(cmd)
                pane.enter()
            else:
                # Make command into a SLURM command
                base_data_dir = config_mgr.get_prop("base_data_dir")
                python_path = osp.join(osp.expanduser("~"), "miniconda3",
                        "envs", conda_env, "bin")
                runs_dir = "data/log/runs"
                if not osp.exists(runs_dir):
                    os.makedirs(runs_dir)

                parts = cmd.split(" ")
                prefix = None
                for i,x in enumerate(parts):
                    if x == '--prefix' or x == 'PREFIX':
                        prefix = parts[i+1].replace('"','')
                        break

                new_log_dir = osp.join(base_data_dir, 'log')
                new_vids_dir = osp.join(base_data_dir, 'vids')
                new_save_dir = osp.join(base_data_dir, 'trained_models')
                ident = str(uuid.uuid4())[:8]
                log_file = osp.join(runs_dir, ident) + ".log"

                last_pane = new_window.attached_pane
                last_pane.send_keys(f"tail -f {log_file}", enter=False)
                pane = new_window.split_window(attach=False)
                pane.set_height(height=10)

                if 'habitat_baselines.run' in cmd:
                    run_file = generate_hab_run_file(new_log_dir, new_vids_dir,
                            new_save_dir, log_file, ident, python_path, cmd,
                            prefix, args)
                    print(f"Running file at {run_file}")
                    pane.send_keys(f"sbatch {run_file}")
                else:
                    cmd += f" --log-dir {new_log_dir}"
                    cmd += f" --vid-dir {new_vids_dir}"
                    cmd += f" --save-dir {new_save_dir}"
                    srun_settings = f"--gres=gpu:{args.g} " + \
                            f"-p {args.st} " + \
                            f"-c {args.c} " + \
                            f"-J {prefix}_{ident} " + \
                            f"-o {log_file}"

                    # This assumes the command begins with "python ..."
                    cmd = f"srun {srun_settings} {python_path}/{cmd}"
                    pane.send_keys(cmd)

        print('everything should be running...')

def generate_hab_run_file(log_dir, vids_dir, save_dir, log_file, ident,
        python_path, cmd, prefix, args):
    fcontents = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --gres gpu:%i
#SBATCH --nodes %i
#SBATCH --cpus-per-task %i
#SBATCH --ntasks-per-node 1
#SBATCH -p %s

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
srun %s/%s"""
    job_name = prefix + '_' + ident
    log_file_loc = '/'.join(log_file.split('/')[:-1])
    fcontents = fcontents % (job_name, log_file, args.g, args.nodes, args.c,
            args.st, python_path, cmd)
    job_file = osp.join(log_file_loc, job_name + '.sh')
    with open(job_file, 'w') as f:
        f.write(fcontents)
    return job_file



def full_execute_command_file():
    parser = get_arg_parser()
    print(os.getcwd())
    args, rest = parser.parse_known_args()
    config_mgr.init(args.cfg)

    cmd_path = osp.join(config_mgr.get_prop('cmds_loc'), args.cmd)
    execute_command_file(cmd_path, rest, args.cd, args.sess_name, args.sess_id,
            args.seed, args)

if __name__ == '__main__':
    full_execute_command_file()

