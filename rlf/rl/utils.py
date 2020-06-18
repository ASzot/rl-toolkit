import glob
import os
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import moviepy.editor as mpy
import gym
import matplotlib.pyplot as plt
import pickle

from contextlib import contextmanager
from timeit import default_timer
from collections import defaultdict
from PIL import Image
import cv2
import hashlib

try:
    import wandb
except:
    pass

def plot_line(plot_vals, save_name, args, to_wb, update_iter=None):
    """
    Plot a simple rough line.
    """
    save_path = osp.join(args.save_dir, args.env_name, args.prefix, save_name)
    plt.title(save_name)
    plt.plot(np.arange(len(plot_vals)), plot_vals)
    plt_save(save_path)

    kwargs = {}
    if update_iter is not None:
        kwargs['step'] = update_iter

    if to_wb:
        wandb.log({save_name:
            [wandb.Image(Image.open(save_path))]}, **kwargs)

def plt_save(*path_parts):
    save_name = osp.join(*path_parts)
    save_dir = osp.dirname(save_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_name)
    print(f"Saved fig to {save_name}")
    plt.clf()


def save_model(model, save_name, args):
    save_dir = osp.join(args.save_dir, args.env_name, args.prefix)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


@contextmanager
def elapsed_timer():
    """
    Measure time elapsed in a block of code. Used for debugging.
    Taken from:
    https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def human_format_int(num, round_pos=2):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    format_str = "%." + str(round_pos) + "f"

    num_str = format_str % num
    num_str = num_str.rstrip('0').rstrip('.')

    return num_str + ['', 'K', 'M', 'G', 'T', 'P'][magnitude]

def pstart_sep():
    print('')
    print('-' * 30)

def pend_sep():
    print('-' * 10)
    print('')

def flatten_obs_dict(ob_shape, keep_keys):
    total_dim = 0
    low_val = None
    high_val = None
    for k in keep_keys:
        sub_space = ob_shape.spaces[k]
        assert len(sub_space.shape) == 1
        if low_val is None:
            low_val = sub_space.low.reshape(-1)[0]
            high_val = sub_space.high.reshape(-1)[0]
        else:
            low_val = min(sub_space.low.reshape(-1)[0], low_val)
            high_val = max(sub_space.high.reshape(-1)[0], high_val)
        total_dim += sub_space.shape[0]
    return gym.spaces.Box(shape=(total_dim,),
            low=np.float32(low_val),
            high=np.float32(high_val),
            dtype=np.float32)

def is_dict_obs(ob_space):
    return isinstance(ob_space, gym.spaces.Dict)

def get_ob_keys(ob_space):
    if isinstance(ob_space, gym.spaces.Dict):
        return list(ob_space.spaces.keys())
    else:
        return [None]

def ob_to_np(obs):
    if isinstance(obs, dict):
        for k in obs:
            obs[k] = obs[k].cpu().numpy()
        return obs
    else:
        return obs.cpu().numpy()

def clone_ob(obs):
    if isinstance(obs, dict):
        return {k: np.array(v) for k,v in obs.items()}
    return np.array(obs)

def ob_to_tensor(obs, device):
    if isinstance(obs, dict):
        for k in obs:
            obs[k] = torch.tensor(obs[k]).to(device)
        return obs
    else:
        return torch.tensor(obs).to(device)

def ob_to_cpu(obs):
    new_obs = {}
    if isinstance(obs, dict):
        for k in obs:
            new_obs[k] = obs[k].cpu()
        return new_obs
    elif obs is None:
        return None
    else:
        return obs.cpu()

def ac_space_to_tensor(action_space):
    return torch.tensor(action_space.low), torch.tensor(action_space.high)

def multi_dim_clip(val, low, high):
    return torch.max(torch.min(val, high), low)

def get_ob_shapes(ob_space):
    if isinstance(ob_space, gym.spaces.Dict):
        return {k: space.shape for k, space in ob_space.spaces.items()}
    else:
        return {None: ob_space.shape}

def get_ob_shape(obs_space, k):
    if k is None:
        return obs_space.shape
    else:
        return obs_space.spaces[k].shape

def get_obs_shape(ob_space):
    if isinstance(ob_space, gym.spaces.Dict):
        return ob_space.spaces['observation'].shape
    else:
        return ob_space.shape

def get_obs_space(ob_space):
    if isinstance(ob_space, gym.spaces.Dict):
        return ob_space.spaces['observation']
    else:
        return ob_space

def get_def_obs(obs):
    if isinstance(obs, dict):
        return obs['observation']
    else:
        return obs

def obs_select(obs, idx):
    if isinstance(obs, dict):
        return {k: obs[k][idx] for k in obs}
    return obs[idx]

def deep_get_other_obs(obs):
    return [get_other_obs(o) for o in obs]

def deep_get_def_obs(obs):
    return [get_def_obs(o) for o in obs]

def get_other_obs(obs):
    if isinstance(obs, dict):
        return {k: obs[k] for k in obs if k != 'observation'}
    else:
        return {}

def combine_spaces(orig_space, new_space_key, new_space):
    if isinstance(orig_space , gym.spaces.Dict):
        return gym.spaces.Dict({
            **orig_space.spaces,
            new_space_key: new_space,
            })
    else:
        return gym.spaces.Dict({
            'observation': orig_space,
            new_space_key: new_space,
            })


def combine_obs(orig_obs, new_obs_key, new_obs):
    if isinstance(orig_obs, dict):
        return {
                **orig_obs,
                new_obs_key: new_obs
                }
    else:
        return {
                'observation': orig_obs,
                new_obs_key: new_obs
                }


def reshape_obs_space(obs_space, new_shape):
    assert isinstance(obs_space, gym.spaces.Box)
    return gym.spaces.Box(
            shape=new_shape,
            high=obs_space.low.reshape(-1)[0],
            low=obs_space.high.reshape(-1)[0],
            dtype=obs_space.dtype)




def get_ac_repr(ac, action):
    """
    Either returns the continuous value of the action or the one-hot encoded
    action
    """
    if isinstance(ac, gym.spaces.Box):
        return action
    elif isinstance(ac, gym.spaces.Discrete):
        y_onehot = torch.zeros(action.shape[0], ac.n).to(action.device)
        y_onehot = y_onehot.scatter(1, action.long(), 1)
        return y_onehot
    else:
        raise ValueError('Invalid action space type')

def get_ac_compact(ac, action):
    """
    Returns the opposite of `get_ac_repr`
    """
    if isinstance(ac, gym.spaces.Box):
        return action
    elif isinstance(ac, gym.spaces.Discrete):
        return torch.argmax(action, dim=-1).unsqueeze(-1)
    else:
        raise ValueError('Invalid action space type')


def get_ac_dim(ac):
    """
    Returns the dimensionality of the action space
    """
    if isinstance(ac, gym.spaces.Box):
        return ac.shape[0]
    elif isinstance(ac, gym.spaces.Discrete):
        return ac.n
    else:
        raise ValueError('Invalid action space type')

def agg_ep_log_stats(env_infos, alg_info):
    """
    Combine the values we want to log into one dictionary for logging.
    - env_info: (list[dict]) returns everything starting with 'ep_' and everything
      in the 'episode' key. There is a list of dicts for each environment
      process.
    - alg_info (dict) returns everything starting with 'alg_add_'
    """

    for k in alg_info:
        if k.startswith('alg_add_'):
            all_log_stats[k].append(inf[k])

    all_log_stats = defaultdict(list)
    for inf in env_infos:
        if 'episode' in inf:
            # Only log at the end of the episode
            for k in inf:
                if k.startswith('ep_'):
                    all_log_stats[k].append(inf[k])
            for k, v in inf['episode'].items():
                all_log_stats[k].append(v)
    return all_log_stats


# Get a render frame function (Mainly for transition)
def get_render_frame_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].unwrapped.render_frame
    elif hasattr(venv, 'venv'):
        return get_render_frame_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_frame_func(venv.env)

    return None

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def save_mp4(frames, vid_dir, name, fps=60.0, no_frame_drop=False):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    vid_file = osp.join(vid_dir, name + '.mp4')
    if osp.exists(vid_file):
        os.remove(vid_file)

    if no_frame_drop:
        def f(t):
            idx = min(int(t * fps), len(frames)-1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames)/fps)
        video.write_videofile(vid_file, fps, verbose=False)

    else:
        drop_frame = 1.5
        def f(t):
            frame_length = len(frames)
            new_fps = 1./(1./fps + 1./frame_length)
            idx = min(int(t*new_fps), frame_length-1)
            return frames[int(drop_frame*idx)]
        video = mpy.VideoClip(f, duration=len(frames)/fps/drop_frame)
        video.write_videofile(vid_file, fps, verbose=False)
    print(f"Rendered to {vid_file}")

def render_text(frame, txt, line):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = frame.shape[0] / 500 / 2
    line_type = 1
    padding = int(frame.shape[0] * 0.05 / 2)
    text_width, text_height = cv2.getTextSize(txt, font, scale, line_type)[0]
    line_offset = line * (text_height + padding)
    frame = frame.astype(np.uint8)
    cv2.putText(frame, txt, (padding,line_offset+padding+text_height),
            font, scale, (255,255,255), line_type)
    return frame

def update_args(args, update_dict, check_exist=False):
    args_dict = vars(args)
    for k, v in update_dict.items():
        if check_exist and k not in args_dict:
            raise ValueError(f"Could not set key {k}")
        args_dict[k] = v


CACHE_PATH = './data/cache'
class CacheHelper:
    def __init__(self, cache_name, lookup_val):
        if not osp.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        sec_hash = hashlib.md5(str(lookup_val).encode('utf-8')).hexdigest()
        cache_id = f"{cache_name}_{sec_hash}.pickle"
        self.cache_id = osp.join(CACHE_PATH, cache_id)

    def exists(self):
        return osp.exists(self.cache_id)

    def load(self):
        with open(self.cache_id, 'rb') as f:
            print(f"Loaded cache path {self.cache_id}")
            return pickle.load(f)

    def save(self, val):
        with open(self.cache_id, 'wb') as f:
            pickle.dump(val, f)
        print(f"Cached to {self.cache_id}")


