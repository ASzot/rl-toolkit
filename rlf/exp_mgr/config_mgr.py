import yaml
import os.path as osp

g_settings = None

def get_cached_settings():
    return g_settings

def get_prop(name):
    return get_cached_settings()[name]

def init(cfg_path):
    global g_settings
    with open(cfg_path) as f:
        g_settings = yaml.load(f, Loader=yaml.BaseLoader)

