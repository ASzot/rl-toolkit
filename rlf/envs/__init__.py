from gym.envs.registration import register
from rlf.envs.bit_flip import BIT_FLIP_ID

register(
    id=BIT_FLIP_ID,
    entry_point="tests.dev.her.bit_flip_env:BitFlipEnv",
)


register(
    id="RltPointMassEnvSpawnRange-v0",
    entry_point="rlf.envs.pointmass:PointMassEnvSpawnRange",
)
