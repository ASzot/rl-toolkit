import os.path as osp
import os
from six.moves import shlex_quote
from rlf.rl import utils
import sys
import pipes
import time
import numpy as np
import random
import datetime
import string
import copy
from rlf.exp_mgr import config_mgr
from rlf.rl.loggers.base_logger import BaseLogger

from collections import deque, defaultdict

# Does not necessarily have WB installed
try:
    import wandb
except:
    pass

# Does not necessarily have Ray installed
try:
    from ray.tune.logger import DEFAULT_LOGGERS
    from ray.tune.integration.wandb import WandbLogger
except:
    pass

def get_wb_ray_kwargs():
    return {
            "loggers": DEFAULT_LOGGERS+(WandbLogger, )
            }

def get_wb_ray_config(config):
    config["wandb"] = {
            "project": config_mgr.get_prop("proj_name"),
            "api_key": config_mgr.get_prop("wb_api_key"),
            "log_config": True,
            }
    return config

class WbLogger(BaseLogger):
    """
    Logger for logging to the weights and W&B online service.
    """

    def __init__(self, wb_proj_name=None, should_log_vids=False,
            wb_entity=None, skip_create_wb=False):
        """
        - wb_proj_name: (string) if None, will use the proj_name provided in
          the `config.yaml` file.
        """
        super().__init__()
        if wb_proj_name is None:
            wb_proj_name = config_mgr.get_prop('proj_name')
        if wb_entity is None:
            wb_entity = config_mgr.get_prop('wb_entity')
        self.wb_proj_name = wb_proj_name
        self.wb_entity = wb_entity
        self.should_log_vids = should_log_vids
        self.skip_create_wb = skip_create_wb

    def init(self, args, mod_prefix=lambda x: x):
        super().init(args, mod_prefix)
        if self.skip_create_wb:
            return
        self.wandb = self._create_wandb(args)

    def log_vals(self, key_vals, step_count):
        if self.is_closed:
            return
        try:
            wandb.log(key_vals, step=int(step_count))
        except:
            self.is_closed = True
            print('Wb logger was closed when trying to log')

    def watch_model(self, model):
        wandb.watch(model)

    def log_image(self, k, img_file, step_count):
        wandb.log({k: wandb.Image(img_file)}, step=step_count)

    def _create_wandb(self, args):
        args.prefix = self.prefix
        if self.prefix.count('-') >= 4:
            # Remove the seed and random ID info.
            parts = self.prefix.split('-')
            group_id = '-'.join([*parts[:2], *parts[4:]])
        else:
            group_id = None

        self.run = wandb.init(project=self.wb_proj_name, name=self.prefix,
                entity=self.wb_entity, group=group_id, reinit=True)
        wandb.config.update(args)
        self.is_closed = False
        return wandb

    def log_video(self, video_file, step_count, fps):
        if not self.should_log_vids:
            return
        wandb.log({'video': wandb.Video(video_file + '.mp4', fps=fps)}, step=step_count)

    def close(self):
        self.is_closed = True
        self.run.finish()


