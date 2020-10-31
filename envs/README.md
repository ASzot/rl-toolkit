Automated installation for popular environments. 

# Installing Environments
Instructions for common environments. Sometimes it's as easy as pip install. 
- `gym-minigrid`: 
  - `pip install gym-minigrid`
  - Environment wrappers for flat and image states at
    `rlf/envs/minigrid_interface.py`.
- `dm_control`
  - `pip install dm_control` 
  - Also be sure you have `export MJLIB_PATH=~/.mujoco/mujoco200/bin/libmujoco200.so` (or
    wherever your MuJoCo install is).
  - To run these environments use format `--env-name dm.domain.task`
- `robosuite`
  - `pip install robosuite`


