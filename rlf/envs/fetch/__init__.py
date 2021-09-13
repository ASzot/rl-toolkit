import rlf.envs.fetch.fetch_interface
from gym.envs.registration import register

register(
    id="VizFetchPickAndPlaceCustom-v0",
    entry_point="rlf.envs.fetch.pick:VizFetchPickAndPlaceEnv",
    max_episode_steps=50,
)

register(
    id="FetchPushNoise-v0",
    entry_point="rlf.envs.fetch.push:FetchPushNoise",
    max_episode_steps=60,
)


register(
    id="FetchPickAndPlaceNoise-v0",
    entry_point="rlf.envs.fetch.pick:FetchPickAndPlaceNoise",
    max_episode_steps=50,
)
