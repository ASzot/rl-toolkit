import os
import os.path as osp

import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env
from rlf.envs.fetch.base_fetch import FetchEnv


class VizFetchPushEnv(FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MODEL_XML_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "viz_push.xml")
        FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=[0.0, 0, -0.3],
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self)
        self.max_episode_steps = 60

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = [1.34193362, 0.74910034, 0.6]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 0.8
        # self.viewer.cam.azimuth = 132.
        self.viewer.cam.azimuth = 165
        self.viewer.cam.elevation = -2


Y_NOISE = 0.02
X_NOISE = 0.05
OBJ_X_NOISE = 0.05
OFFSET = 0.10


class FetchPushNoise(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.set_noise_ratio(1.0, 1.0)
        MODEL_XML_PATH = os.path.join("fetch", "push.xml")
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0,
            # The ranges shouldn't matter because we sample ourselves
            obj_range=0.1,
            target_range=0,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self)

    def set_noise_ratio(self, noise_ratio, goal_noise_ratio):
        self.obj_low = [-noise_ratio * OBJ_X_NOISE, 0]
        self.obj_high = [noise_ratio * OBJ_X_NOISE, noise_ratio * Y_NOISE * 2]

        self.goal_low = [-goal_noise_ratio * X_NOISE, -goal_noise_ratio * Y_NOISE * 2]
        self.goal_high = [goal_noise_ratio * X_NOISE, 0]

    def _get_obs(self):
        obs = super()._get_obs()
        obs["observation"] = np.concatenate([obs["observation"], obs["desired_goal"]])
        return obs

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2] + np.array([0.0, OFFSET])
            object_xpos += self.np_random.uniform(self.obj_low, self.obj_high)

            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + np.array([0.0, -1 * OFFSET, 0.0])
        goal[:2] += self.np_random.uniform(self.goal_low, self.goal_high)

        goal += self.target_offset
        goal[2] = self.height_offset
        return goal.copy()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        lookat = [1.34193362, 0.74910034, 0.55472272]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.3
        self.viewer.cam.azimuth = 132
        self.viewer.cam.elevation = -14.0

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()
