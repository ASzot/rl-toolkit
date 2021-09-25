import gym
import torch


class D4rlDataset:
    def __init__(self, env_name, traj_path):
        env = gym.make(env_name)
        self.dataset = env.get_dataset()

    def convert_to_override_data(self):
        dones = self.dataset["timeouts"] | self.dataset["terminals"]
        use_obs = self.dataset["observations"]
        dones = dones[:-1]
        obs = use_obs[:-1]
        next_obs = use_obs[1:]
        actions = self.dataset["actions"][:-1]
        return {
            "done": torch.FloatTensor(dones),
            "obs": torch.tensor(obs),
            "next_obs": torch.tensor(next_obs),
            "actions": torch.tensor(actions),
        }
