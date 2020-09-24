import os.path as osp
import os
from six.moves import shlex_quote
import sys
import pipes
import time
import numpy as np
import random
import datetime
import string
from rlf.exp_mgr import config_mgr

from collections import deque, defaultdict


class BaseLogger(object):
    def __init__(self):
        pass

    def init(self, args):
        self.is_debug_mode = args.prefix == 'debug'
        self._create_prefix(args)

        print('Smooth len is %i' % args.log_smooth_len)

        self._step_log_info = defaultdict(
            lambda: deque(maxlen=args.log_smooth_len))

        if not self.is_debug_mode:
            self.save_run_info(args)
        else:
            print('In debug mode')
        self.is_printing = True
        self.prev_steps = 0

    def disable_print(self):
        self.is_printing = False

    def save_run_info(self, args):
        log_dir = osp.join(args.log_dir, args.env_name, self.prefix)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        # cmd
        train_cmd = 'python3 main.py ' + \
            ' '.join([pipes.quote(s) for s in sys.argv[1:]])
        with open(osp.join(log_dir, "cmd.txt"), "a+") as f:
            f.write(train_cmd)

        # git diff
        print('Save git commit and diff to {}/git.txt'.format(log_dir))
        cmds = ["echo `git rev-parse HEAD` >> {}".format(
            shlex_quote(osp.join(log_dir, 'git.txt'))),
            "git diff >> {}".format(
            shlex_quote(osp.join(log_dir, 'git.txt')))]
        os.system("\n".join(cmds))

        args_lines = "Date and Time:\n"
        args_lines += time.strftime("%d/%m/%Y\n")
        args_lines += time.strftime("%H:%M:%S\n\n")
        arg_dict = args.__dict__
        for k in sorted(arg_dict.keys()):
            args_lines += "{}: {}\n".format(k, arg_dict[k])

        with open(osp.join(log_dir, "args.txt"), "w") as f:
            f.write(args_lines)

    def backup(self, args, global_step):
        log_dir = osp.join(args.log_dir, args.env_name, args.prefix)
        model_dir = osp.join(args.save_dir, args.env_name, args.prefix)
        vid_dir = osp.join(args.vid_dir, args.env_name, args.prefix)

        log_base_dir = log_dir.rsplit('/', 1)[0]
        model_base_dir = model_dir.rsplit('/', 1)[0]
        vid_base_dir = vid_dir.rsplit('/', 1)[0]
        proj_name = config_mgr.get_prop('proj_name')
        sync_host = config_mgr.get_prop('sync_host')
        sync_user = config_mgr.get_prop('sync_user')
        sync_port = config_mgr.get_prop('sync_port')
        cmds = [
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                sync_port, sync_user, sync_host, proj_name, log_dir),
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                sync_port, sync_user, sync_host, proj_name, model_dir),
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                sync_port, sync_user, sync_host, proj_name, vid_dir),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                sync_port, log_dir, sync_user, sync_host, proj_name, log_base_dir),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                sync_port, model_dir, sync_user, sync_host, proj_name, model_base_dir),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                sync_port, vid_dir, sync_user, sync_host, proj_name, vid_base_dir),
        ]
        os.system("\n".join(cmds))
        print('\n' + '*' * 50)
        print('*' * 5 + ' backup at global step {}'.format(global_step))
        print('*' * 50 + '\n')
        print('')

    def collect_step_info(self, step_log_info):
        for k in step_log_info:
            self._step_log_info[k].extend(step_log_info[k])

    def _get_env_id(self, args):
        upper_case = [c for c in args.env_name if c.isupper()]
        if len(upper_case) == 0:
            return ''.join([word[0] for word in args.env_name.split(".")])
        else:
            return ''.join(upper_case)

    def _create_prefix(self, args):
        assert args.prefix is not None and args.prefix != '', 'Must specify a prefix'
        d = datetime.datetime.today()
        date_id = '%i%i' % (d.month, d.day)
        env_id = self._get_env_id(args)
        rnd_id = ''.join(random.sample(
            string.ascii_uppercase + string.digits, k=2))
        before = ('%s-%s-%s-%s-' %
                  (date_id, env_id, args.seed, rnd_id))

        if args.prefix != 'debug' and args.prefix != 'NONE':
            self.prefix = before + args.prefix
            print('Assigning full prefix %s' % self.prefix)
        else:
            self.prefix = args.prefix

    def set_prefix(self, args):
        args.prefix = self.prefix

    def start_interval_log(self):
        self.start = time.time()

    def log_vals(self, key_vals, step_count):
        """
        Log key value pairs to whatever interface.
        """
        pass

    def log_video(self, video_file, step_count, fps):
        pass

    def watch_model(self, model):
        """
        - model (torch.nn.Module) the set of parameters to watch
        """
        pass

    def interval_log(self, j, total_num_steps, episode_count, updater_log_vals, args):
        end = time.time()

        fps = int((total_num_steps - self.prev_steps) / (end - self.start))
        self.prev_steps = total_num_steps
        self.num_frames = 0
        num_eps = len(self._step_log_info.get('r', []))
        rewards = self._step_log_info.get('r', [0])

        log_stat_vals = {}
        for k, v in self._step_log_info.items():
            log_stat_vals['avg_' + k] = np.mean(v)
            log_stat_vals['min_' + k] = np.min(v)
            log_stat_vals['max_' + k] = np.max(v)

        def should_print(x):
            return '_pr_' in x

        log_dat = {
                **updater_log_vals,
                **log_stat_vals,
            }

        if self.is_printing:
            print(f"Updates {j}, Steps {total_num_steps}, Episodes {episode_count}, FPS {fps}")
            if args.num_steps != 0:
                print(
                    f"Over the last {num_eps} episodes:\n"
                    f"mean/median reward {np.mean(rewards):.2f}/{np.median(rewards)}\n"
                    f"min/max {np.min(rewards):.2f}/{np.max(rewards):.2f}"
                )

            # Print log values from the updater if requested.
            for k, v in log_dat.items():
                if should_print(k):
                    print(f"    - {k}: {v}")

            # Print a new line to separate loggin lines and keep things clean.
            print('')
            print('')

        # Log all values
        log_dat['fps'] = fps
        self.log_vals(log_dat, total_num_steps)
        return log_dat

    def close(self):
        pass
