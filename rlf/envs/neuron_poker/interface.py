from rlf.envs.env_interface import EnvInterface, register_env_interface
from rlf.envs.neuron_poker.agent_random import Player
from rlf.args import str2bool
import numpy as np
import rlf.rl.utils as rutils
from rlf.envs.neuron_poker.env import HoldemTable
import gym


class PokerAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                shape=env.observation_space, dtype=np.float32)

    def step(self, a):
        obs, reward, done, info = super().step(a)
        #print(info['player_data'])
        return obs, reward, done, info


class PokerInterface(EnvInterface):
    def get_add_args(self, parser):
        parser.add_argument('--poker-n-players', type=int, default=2, help="""
                Including the agent.
                """)

    def create_from_id(self, env_id):
        stack = 500
        env = HoldemTable(initial_stacks=stack)
        for _ in range(self.args.poker_n_players-1):
            player = Player()
            env.add_player(player)

        player = Player()
        player.autoplay = False
        env.add_player(player)
        env.reset()

        env = PokerAdapter(env)

        return env

register_env_interface("Poker-v0", PokerInterface)
