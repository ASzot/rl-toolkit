import gym
import numpy as np
import torch
from gym import spaces
from rlf import EnvInterface, register_env_interface
from rlf.args import str2bool
from rlf.baselines.vec_env.vec_env import VecEnv
from torch.distributions import Uniform

VEL_LIMIT = 10.0
POS_LIMIT = 1.5
ERROR_MSG = (
    "Make sure rlf.envs.pointmass is imported. Also make sure multiprocessing is forced"
)


class PointMassEnvSpawnRange(gym.Env):
    """
    You have to use the batched version, this is just a dummy class for the gym registration.
    """

    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def step(self, action):
        raise NotImplementedError(ERROR_MSG)

    def reset(self):
        raise NotImplementedError(ERROR_MSG)


class BatchedTorchPointMassEnvSpawnRange(VecEnv):
    def __init__(self, args, set_eval, obs_space=None):
        self._batch_size = args.num_processes
        self._is_eval = set_eval or args.pm_force_eval_start_dist
        if args.pm_force_train_start_dist:
            self._is_eval = False
        self.args = args

        self.pos_dim = 2
        self._device = args.device
        self._goal = torch.tensor([0.0, 0.0]).to(self._device)
        self._ep_step = 0

        self._ep_rewards = []
        if obs_space is None:
            obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        super().__init__(
            self._batch_size,
            obs_space,
            spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        )

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def forward(self, cur_pos, cur_vel, action):
        action = torch.clamp(action, -1.0, 1.0)
        new_vel = cur_vel
        new_pos = cur_pos + (action * self.args.pm_dt)

        if self.args.pm_clip:
            new_pos = torch.clamp(new_pos, -POS_LIMIT, POS_LIMIT)
        return new_pos, new_vel

    def step(self, action):
        self.cur_pos, self.cur_vel = self.forward(self.cur_pos, self.cur_vel, action)
        self._ep_step += 1

        is_done = self._ep_step >= self.args.pm_ep_horizon
        dist_to_goal = torch.linalg.norm(
            self._goal - self.cur_pos, dim=-1, keepdims=True
        )

        reward = -self.args.pm_reward_dist_pen * dist_to_goal
        self._ep_rewards.append(reward)

        all_is_done = [is_done for _ in range(self._batch_size)]

        all_info = [
            {"ep_dist_to_goal": dist_to_goal[i].item()} for i in range(self._batch_size)
        ]

        if is_done:
            final_obs = self._get_obs()
            for i in range(self._batch_size):
                all_info[i]["episode"] = {
                    "r": torch.stack(self._ep_rewards).sum(0)[i].item()
                }
                all_info[i]["final_obs"] = final_obs[i]
            self.reset()

        return (self._get_obs(), reward, all_is_done, all_info)

    @staticmethod
    def get_regions(offset, spread):
        inc = np.pi / 2

        centers = [offset + i * inc for i in range(4)]

        return torch.tensor([[center - spread, center + spread] for center in centers])

    def _sample_start(self, batch_size):
        if self._is_eval:
            idx = torch.randint(0, 4, (batch_size,))
            regions = BatchedTorchPointMassEnvSpawnRange.get_regions(
                0.0, self.args.pm_start_state_noise
            )
        else:
            idx = torch.randint(0, self.args.pm_num_train_regions, (batch_size,))
            regions = BatchedTorchPointMassEnvSpawnRange.get_regions(
                np.pi / 4, self.args.pm_start_state_noise
            )

        if torch.isclose(regions[idx, 0], regions[idx, 1]).all():
            ang = regions[idx, 0]
        else:
            ang = Uniform(regions[idx, 0], regions[idx, 1]).sample()
        radius = np.sqrt(2)
        return (
            torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
            + self._goal
        )

    def reset(self):
        self.cur_pos = self._sample_start(self._batch_size)
        self.cur_vel = torch.zeros(self._batch_size, 2)
        self._ep_step = 0
        self._ep_rewards = []

        return self._get_obs()

    def _get_obs(self):
        return self.cur_pos.clone()


class BatchedTorchPointMassEnvSingleSpawn(BatchedTorchPointMassEnvSpawnRange):
    def reset(self):
        super().reset()
        # Points must move clockwise starting from quadrant 1.
        all_start = torch.tensor(
            [
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, -1.0],
            ]
        ).view(-1, 1, 2)
        self.cur_vel = torch.zeros(self._batch_size, 2)
        self.cur_pos = all_start[self.args.pm_start_idx].repeat(self._batch_size, 1)

        self.start_pos = self.cur_pos.clone()
        return self._get_obs()


class PointMassInterface(EnvInterface):
    def requires_tensor_wrap(self) -> bool:
        return False

    def get_setup_multiproc_fn(
        self,
        make_env,
        env_id,
        seed,
        allow_early_resets,
        env_interface,
        set_eval,
        alg_env_settings,
        args,
    ):
        if args.pm_start_idx >= 0:
            return BatchedTorchPointMassEnvSingleSpawn(args, set_eval)
        else:
            return BatchedTorchPointMassEnvSpawnRange(args, set_eval)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(
            "--pm-force-eval-start-dist",
            type=str2bool,
            default=False,
            help="""
            If true, using the EVAL start state dist even during TRAINING.
            """,
        )
        parser.add_argument(
            "--pm-force-train-start-dist",
            type=str2bool,
            default=False,
            help="""
            If true, using the TRAINING start state dist even during EVAL.
            """,
        )
        parser.add_argument(
            "--pm-clip",
            type=str2bool,
            default=True,
            help="""
                If true, clips the agent to a region
                """,
        )
        parser.add_argument(
            "--pm-ep-horizon",
            type=int,
            default=5,
            help="""
                Controls how long each episode is.
                """,
        )
        parser.add_argument(
            "--pm-num-train-regions",
            type=int,
            default=4,
            help="""
                Controls how many regions to sample from during TRAINING.
                """,
        )
        parser.add_argument(
            "--pm-start-state-noise",
            type=float,
            default=np.pi / 20,
            help="""
                Sets the amount of starting state noise.
                """,
        )
        parser.add_argument(
            "--pm-dt",
            type=float,
            default=0.2,
            help="The time step. The higher, the larger of a step",
        )
        parser.add_argument(
            "--pm-reward-dist-pen",
            type=float,
            default=1 / 10.0,
            help="""
            The scaling factor on the distance to goal penalty
            """,
        )

        parser.add_argument(
            "--pm-start-idx",
            type=int,
            default=-1,
            help="""
                If non-negative. This will select one of a pre-set number of
                locations for the starting position.
                """,
        )


register_env_interface("^RltPointMass", PointMassInterface)
