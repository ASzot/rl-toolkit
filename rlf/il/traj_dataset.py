import torch
import numpy as np
import rlf.rl.utils as rutils
from rlf.il.il_dataset import ImitationLearningDataset

class TrajDataset(ImitationLearningDataset):
    """
    See `rlf/il/il_dataset.py` for notes about the demonstration dataset
    format.
    """
    def __init__(self, load_path):
        trajs = torch.load(load_path)

        rutils.pstart_sep()
        self._setup(trajs)

        trajs = self._generate_trajectories(trajs)

        assert len(trajs) != 0, 'No trajectories found to load!'

        self.n_trajs = len(trajs)
        print('Collected %i trajectories' % len(trajs))
        # Compute statistics across the trajectories.
        all_obs = torch.cat([t[0] for t in trajs])
        all_actions = torch.cat([t[1] for t in trajs])
        self.state_mean = torch.mean(all_obs, dim=0)
        self.state_std = torch.std(all_obs, dim=0)
        self.action_mean = torch.mean(all_actions, dim=0)
        self.action_std = torch.std(all_actions, dim=0)

        self.data = self._gen_data(trajs)
        self.traj_lens = [len(traj[0]) for traj in trajs]
        self.trajs = trajs
        self.holdout_idxs = []

        rutils.pend_sep()

    def get_num_trajs(self):
        return self.n_trajs

    def compute_split(self, traj_frac):
        traj_count = int(len(self.trajs) * traj_frac)
        all_idxs = np.arange(0, len(self.trajs))
        np.random.shuffle(all_idxs)
        idxs = all_idxs[:traj_count]
        self.holdout_idxs = all_idxs[traj_count:]
        self.n_trajs = traj_count

        self.data = self._gen_data([self.trajs[i] for i in idxs])
        return self


    def _setup(self, trajs):
        """
        Initialization subclasses need to perform. Cannot perform
        initialization in __init__ as traj is not avaliable.
        """
        pass

    def viz(self, args):
        import seaborn as sns
        sns.distplot(self.traj_lens)
        rutils.plt_save(args.save_dir, args.env_name, args.prefix, 'traj_len_dist.png')

    def get_expert_stats(self, device):
        return {
                'state': (self.state_mean.to(device), self.state_std.to(device)),
                'action': (self.action_mean.to(device), self.action_std.to(device))
                }

    def __getitem__(self, i):
        return self.data[i]

    def _gen_data(self, trajs):
        """
        Can define in inhereted class to perform a custom transformation over
        the trajectories.
        """
        return trajs

    def should_terminate_traj(self, j, obs, next_obs, done, actions):
        return done[j]

    def _generate_trajectories(self, trajs):
        # Get by trajectory instead of transition
        obs_dim = trajs['obs'].shape[1:]

        done = trajs['done'].float()
        obs = trajs['obs'].float()
        next_obs = trajs['next_obs'].float()
        actions = trajs['actions'].float()

        trajs = []

        num_samples = done.shape[0]
        print('Collecting trajectories')
        start_j = 0
        j = 0
        while j < num_samples:
            if self.should_terminate_traj(j, obs, next_obs, done, actions):
                obs_seq = obs[start_j:j+1]
                final_obs = next_obs[j]

                combined_obs = torch.cat([obs_seq, final_obs.view(1, *obs_dim)])

                trajs.append((combined_obs, actions[start_j:j+1]))
                # Move to where this episode ends
                while j < num_samples and not done[j]:
                    j += 1
                start_j = j + 1

            if j < num_samples and done[j]:
                start_j = j + 1

            j += 1
        return trajs

    def __len__(self):
        return len(self.data)

