import argparse
import os.path as osp
import uuid
from functools import partial

import gym
import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.args import str2bool
from rlf.baselines.common.atari_wrappers import WarpFrame
from rlf.envs.fetch.fetch_interface import (BlockGripperActionWrapper,
                                            FetchNoVelWrapper)
from rlf.envs.image_obs_env import ImageObsWrapper
from rlf.exp_mgr.viz_utils import save_agent_obs, save_mp4
from rlf.il import GoalTrajSaver
from rlf.rl.envs import TransposeImage
from tqdm import tqdm


def get_state(obs):
    if "state" in obs:
        return obs["state"]
    else:
        return obs["observation"]


def goToGoal(env, obs):
    episode_frames = []
    timeStep = 0  # count the total number of timesteps
    HEIGHT = 0.08

    all_actions = []
    all_obs = [obs]
    all_info = []
    all_frames = []
    all_dones = []

    if hasattr(env, "max_env_steps"):
        max_ep_steps = env.max_episode_steps
    else:
        max_ep_steps = env._max_episode_steps

    def get_info(next_obs, env):
        is_success_fn = rutils.get_env_attr(env, "_is_success")
        is_success = is_success_fn(next_obs["achieved_goal"], next_obs["desired_goal"])
        return {
            "ep_found_goal": float(is_success),
            "final_obs": next_obs["observation"],
        }

    def compute_behind_pos(objectPos, object_rel_pos, goal, gripper_pos):
        to_goal = goal - objectPos
        to_goal /= np.linalg.norm(to_goal)
        to_goal *= 0.1
        to_pos = object_rel_pos - to_goal
        to_pos[2] += HEIGHT
        return to_pos

    def compute_push_pos(objectPos, object_rel_pos, goal, gripper_pos):
        to_goal = goal - gripper_pos
        return to_goal

    def compute_target_pos(objectPos, object_rel_pos, goal, gripper_pos, target_pos):
        return target_pos - gripper_pos

    def move_to(compute_goal, lastObs, timeStep, episode_frames, never_stop=False):
        goal = lastObs["desired_goal"]
        gripper_pos = get_state(lastObs)[0:3]
        objectPos = get_state(lastObs)[3:6]
        object_rel_pos = get_state(lastObs)[6:9]

        relative_pos = compute_goal(objectPos, object_rel_pos, goal, gripper_pos)

        while (
            never_stop or np.linalg.norm(relative_pos) >= 0.005
        ) and timeStep <= max_ep_steps:
            all_frames.append(env.render("rgb_array"))
            action = [0, 0, 0]
            relative_pos = compute_goal(objectPos, object_rel_pos, goal, gripper_pos)

            for i in range(len(relative_pos)):
                action[i] = relative_pos[i] * 6

            lastObs, reward, done, info = env.step(action)
            timeStep += 1

            all_actions.append(action)
            all_info.append(get_info(lastObs, env))
            all_obs.append(lastObs)
            all_dones.append(done)

            gripper_pos = get_state(lastObs)[0:3]
            objectPos = get_state(lastObs)[3:6]
            object_rel_pos = get_state(lastObs)[6:9]
        return lastObs, timeStep

    gripper_pos = get_state(obs)[0:3]
    target_pos = gripper_pos.copy()
    target_pos[2] += HEIGHT
    obs, timeStep = move_to(
        partial(compute_target_pos, target_pos=target_pos),
        obs,
        timeStep,
        episode_frames,
    )

    obs, timeStep = move_to(compute_behind_pos, obs, timeStep, episode_frames)

    gripper_pos = get_state(obs)[0:3]
    target_pos = gripper_pos.copy()
    target_pos[2] -= HEIGHT
    obs, timeStep = move_to(
        partial(compute_target_pos, target_pos=target_pos),
        obs,
        timeStep,
        episode_frames,
    )

    obs, timeStep = move_to(
        compute_push_pos, obs, timeStep, episode_frames, never_stop=True
    )

    return all_frames, all_obs, all_actions, all_info, all_dones


def main(render, count, args):
    rnd_folder = str(uuid.uuid4()).split("-")[0]
    traj_saver = GoalTrajSaver(
        osp.join("./data/traj/", rnd_folder, args.env_name),
        False,
        args.save_name + ".pt",
    )

    env = gym.make(args.env_name)
    env = BlockGripperActionWrapper(env)
    if args.easy_obs:
        env = FetchNoVelWrapper(env, False)
    if args.img_dim is not None:
        env = ImageObsWrapper(env, args.img_dim)
        env = WarpFrame(env, grayscale=True)
        keys = rutils.get_ob_keys(env.observation_space)
        transpose_keys = [
            k for k in keys if len(rutils.get_ob_shape(env.observation_space, k)) == 3
        ]
        env = TransposeImage(env, op=[2, 0, 1], transpose_keys=transpose_keys)
        # env = SingleFrameStack(env, 4, "observation")

    all_frames = []
    cur_count = 0
    t = tqdm(total=count)
    while cur_count < count:
        obs = env.reset()
        episode_frames, traj_obs, traj_action, traj_info, traj_done = goToGoal(env, obs)

        traj_done = torch.tensor([1.0 if done else 0.0 for done in traj_done])
        traj_obs = torch.tensor([obs["observation"] for obs in traj_obs])
        traj_action = torch.tensor(traj_action)

        all_frames.extend(episode_frames)
        traj_len = len(traj_done)
        add_count = 0
        for i in range(traj_len):
            add_count += traj_saver.collect(
                traj_obs[i].unsqueeze(0),
                traj_obs[i + 1].unsqueeze(0),
                traj_done[i].unsqueeze(0),
                traj_action[i].unsqueeze(0),
                [traj_info[i]],
            )
        t.update(add_count)
        cur_count += add_count

        if render:
            all_frames.extend(episode_frames)

    if render:
        viz_folder = "./data/vids/viz"
        save_mp4(all_frames, viz_folder, "push_debug", fps=30, no_frame_drop=True)
        raise ValueError()

    t.close()
    traj_saver.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--easy-obs", type=str2bool, default=True)
    parser.add_argument("--env-name", type=str, default="FetchPushNoise-v0")
    parser.add_argument("--save-name", type=str, default="trajs")
    parser.add_argument("--img-dim", type=int, default=None)
    args = parser.parse_args()
    main(args.render, args.count, args)
