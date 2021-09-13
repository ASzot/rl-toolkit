import os
import os.path as osp

import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env

MODEL_XML_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "viz_pick_and_place.xml")
Y_NOISE = 0.02
X_NOISE = 0.05
OBJ_X_NOISE = 0.05
OFFSET = 0.1


class VizFetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        # To manually control things for rendering.
        # fetch_env.FetchEnv.__init__(
        #    self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
        #    gripper_extra_height=0.2, target_in_the_air=True,
        #    # You should change the secon coordinate
        #    target_offset=[0.1, -0.1, 0.1],
        #    obj_range=0.15, target_range=0.3, distance_threshold=0.05,
        #    initial_qpos=initial_qpos, reward_type=reward_type)

        # For actual policy eval / training.
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self)
        self.max_episode_steps = 50

    def set_noise_ratio(self, x, y):
        pass

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


class FetchPickAndPlaceNoise(VizFetchPickAndPlaceEnv):
    def __init__(self, reward_type="sparse"):
        self.low = None
        self.set_noise_ratio(1.0, 1.0)
        super().__init__(reward_type)

    def set_noise_ratio(self, noise_ratio, goal_noise_ratio):
        # Lower X and Y coordinates
        self.obj_low = [-noise_ratio * OBJ_X_NOISE, 0]
        self.obj_high = [noise_ratio * OBJ_X_NOISE, noise_ratio * Y_NOISE * 2]

        self.goal_low = [-goal_noise_ratio * X_NOISE, -goal_noise_ratio * Y_NOISE * 2]
        self.goal_high = [goal_noise_ratio * X_NOISE, 0]

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
        goal[2] += 0.15
        return goal.copy()

    def _get_obs(self):
        obs = super()._get_obs()
        obs["observation"] = np.concatenate([obs["observation"], obs["desired_goal"]])
        return obs
