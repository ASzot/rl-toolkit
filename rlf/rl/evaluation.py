from collections import defaultdict

import numpy as np
import torch
import os
import os.path as osp

from rlf.rl import utils
from rlf.rl.envs import make_vec_envs
from rlf.rl.utils import save_mp4
from rlf.policies.base_policy import get_empty_step_info
from rlf.il.traj_mgr import TrajSaver
from tqdm import tqdm
import time
from rlf.rl.envs import get_vec_normalize



def eval_print(env_interface, args, alg_env_settings, policy, vec_norm,
        total_num_steps, mode, eval_envs, log):
    print('Evaluating ' + mode)
    args.evaluation_mode = True
    eval_info, eval_envs = evaluate(args, alg_env_settings, policy, vec_norm,
                                                 env_interface, total_num_steps,
                                                 mode, eval_envs, log, None)

    log.log_vals({'eval_%s_%s' % (mode, k): np.mean(v)
                  for k, v in eval_info.items()}, total_num_steps)
    args.evaluation_mode = False
    return eval_envs


def train_eval(envs, alg_env_settings, policy, args, log,
               total_num_steps, env_interface,
               train_eval_envs):

    vec_norm = get_vec_normalize(envs)

    train_eval_envs = eval_print(env_interface, args, alg_env_settings, policy,
            vec_norm, total_num_steps, 'train', train_eval_envs, log)

    return train_eval_envs


def full_eval(envs, policy, log, checkpointer, env_interface, args,
        alg_env_settings, create_traj_saver_fn):
    vec_norm = get_vec_normalize(envs)

    args.evaluation_mode = True
    ret_info, envs = evaluate(args, alg_env_settings, policy, vec_norm,
                          env_interface, 0, 'final', None, log,
                          create_traj_saver_fn)
    args.evaluation_mode = False
    envs.close()

    return ret_info


def evaluate(args, alg_env_settings, policy, true_vec_norm, env_interface,
        num_steps, mode, eval_envs, log, create_traj_saver_fn):
    if args.eval_num_processes is None:
        num_processes = args.num_processes
    else:
        num_processes = args.eval_num_processes

    if eval_envs is None:
        eval_envs = make_vec_envs(args.env_name, args.seed + num_steps, num_processes,
                                  args.gamma, args.env_log_dir, args.device, True,
                                  env_interface, args,
                                  alg_env_settings, set_eval=True)

    assert get_vec_normalize(
        eval_envs) is None, 'Norm is manually applied'

    if true_vec_norm is not None:
        obfilt = true_vec_norm._obfilt
    else:
        def obfilt(x, update): return x

    eval_episode_rewards = []
    eval_def_stats = defaultdict(list)
    ep_stats = defaultdict(list)

    obs = eval_envs.reset()

    hidden_states = {}
    eval_masks = torch.zeros(num_processes, 1, device=args.device)

    frames = []
    infos = None

    policy.eval()
    if args.eval_save and create_traj_saver_fn is not None:
        traj_saver = create_traj_saver_fn(osp.join(args.traj_dir, args.env_name, args.prefix))
    else:
        assert not args.eval_save, ('Cannot save evaluation without ',
                'specifying the eval saver creator function')

    total_num_eval = num_processes * args.num_eval

    # Measure the number of episodes completed
    pbar = tqdm(total=total_num_eval)
    evaluated_episode_count = 0
    while evaluated_episode_count < total_num_eval:
        step_info = get_empty_step_info()
        with torch.no_grad():
            act_obs = obfilt(utils.ob_to_np(obs), update=False)
            act_obs = utils.ob_to_tensor(act_obs, args.device)

            ac_info = policy.get_action(utils.get_def_obs(act_obs),
                                        utils.get_other_obs(obs),
                                        hidden_states,
                                        eval_masks, step_info)

            hidden_states = ac_info.hxs

        # Observe reward and next obs
        next_obs, _, done, infos = eval_envs.step(ac_info.take_action)
        if args.eval_save:
            finished_count = traj_saver.collect(obs, next_obs, done, ac_info.take_action, infos)
        else:
            finished_count = sum([int(d) for d in done])

        pbar.update(finished_count)
        evaluated_episode_count += finished_count

        cur_frame = None

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=args.device)
        frames.extend(get_render_frames(eval_envs, env_interface, obs, next_obs,
            ac_info.take_action, eval_masks, infos, args, evaluated_episode_count))
        obs = next_obs

        step_log_vals = utils.agg_ep_log_stats(infos, ac_info.extra)
        for k, v in step_log_vals.items():
            ep_stats[k].extend(v)

    pbar.close()
    info = {}
    if args.eval_save:
        traj_saver.save()

    ret_info = {}

    print(" Evaluation using %i episodes:" % len(ep_stats['r']))
    for k, v in ep_stats.items():
        print(' - %s: %.5f' % (k, np.mean(v)))
        ret_info[k] = np.mean(v)

    save_file = save_frames(frames, mode, num_steps, args)
    if save_file is not None:
        log.log_video(save_file, num_steps, args.vid_fps)

    # Switch policy back to train mode
    policy.train()

    return ret_info, eval_envs


def save_frames(frames, mode, num_steps, args):
    if not osp.exists(args.vid_dir):
        os.makedirs(args.vid_dir)

    add = ''
    if args.load_file != '':
        add = args.load_file.split('/')[-2]
        add += '_'

    save_name = '%s%s_%s' % (add,
                             utils.human_format_int(num_steps), mode)

    save_dir = osp.join(args.vid_dir, args.env_name, args.prefix)

    fps = args.vid_fps

    if len(frames) > 0:
        save_mp4(frames, save_dir, save_name,
                 fps=args.vid_fps, no_frame_drop=True)
        return osp.join(save_dir, save_name)
    return None

def get_render_frames(eval_envs, env_interface, obs, next_obs, action, masks, infos,
            args, evaluated_episode_count):
    if args.num_render is not None and (evaluated_episode_count >= args.num_render):
        return []
    add_kwargs = {}
    if args.render_metric:
        add_kwargs = {
                "obs": utils.ob_to_cpu(obs),
                "action": action.cpu(),
                "next_obs": utils.ob_to_cpu(next_obs),
                "info": infos,
                "next_mask": masks.cpu()
                }

    try:
        cur_frame = eval_envs.render(**env_interface.get_render_args(),
                **add_kwargs)
    except EOFError as e:
        print('This problem can likely be fixed by setting --eval-num-processes 1')
        raise e

    if not isinstance(cur_frame, list):
        cur_frame = [cur_frame]
    return cur_frame

