# Env Interfaces

How to setup a new environment interface that registers custom command line arguments and special initialization logic (such as wrappers or passing arguments to the environment object).
```
from rlf.envs.env_interface import EnvInterface, register_env_interface


class NewEnvInterface(EnvInterface):
    def env_trans_fn(self, env, set_eval):
        # Add wrappers to Gym env here.
        return env

    def create_from_id(self, env_id, local_seed):
        # If you want another way to create environments other than gym.make
        return YourCustomGymEnv()

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--ant-noise", type=float, default=0.0)
        parser.add_argument("--ant-cover", type=int, default=100)
        parser.add_argument("--ant-is-expert", action="store_true")


register_env_interface("YourEnvName-v0", NewEnvInterface)
```

# Added Envs
List of custom included environments:
* `RltPointMass-v0`:
* `MultiGoalRltPointMass-v0`:
