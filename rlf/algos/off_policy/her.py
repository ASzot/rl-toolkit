from rlf.storage.transition_storage import TransitionStorage
import torch
import rlf.rl.utils as rutils
import numpy as np

class HerStorage(TransitionStorage):
    """
    Uses the "final" HER strategy which uses the state achieved at the end of
    the trajectory.
    Observation should have format:
        {
        "achieved_goal": tensor
        "desired_goal": tensor
        "observation": tensor
        }
    Arguments are in `OffPolicy`
    """

    def _on_traj_done(self, done_trajs):
        for done_traj in done_trajs:
            for t in range(len(done_traj) - 1):
                state = done_traj[t][0].copy()

                # Is this the final frame?
                if t < len(done_traj) - 1:
                    next_state = done_traj[t+1][0].copy()
                else:
                    # Yes, this is the final frame, there is no next state.
                    # Just an invalid state
                    next_state = state.copy()

                if t == 0:
                    mask = 1.0
                else:
                    mask = done_traj[t-1][2]

                # Augment with the HER style goal.
                if self.args.her_strat == 'future':
                    for k in range(self.args.her_K):
                        # Randomly choose a time step in the future.
                        future_t = np.random.randint(t, len(done_traj) - 1)
                        future_goal = done_traj[future_t+1][0]['achieved_goal']

                        state['desired_goal'] = future_goal
                        next_state['desired_goal'] = future_goal

                        self._push_transition({
                            'action': done_traj[t][1],
                            'state': state,
                            'mask': torch.tensor([mask]),
                            'rnn_hxs': torch.tensor([0]),
                            'reward': torch.tensor([done_traj[t][4]]),
                            'next_state': next_state,
                            'next_mask': torch.tensor([done_traj[t][2]]),
                            'next_rnn_hxs': torch.tensor([0]),
                            })
                elif self.args.her_strat == 'final':
                    final_goal = done_traj[-1][0]['achieved_goal']
                    state['desired_goal'] = final_goal
                    next_state['desired_goal'] = final_goal
                    self._push_transition({
                            'action': done_traj[t][1],
                            'state': state,
                            'mask': torch.tensor([mask]),
                            'rnn_hxs': torch.tensor([0]),
                            'reward': torch.tensor([done_traj[t][4]]),
                            'next_state': next_state,
                            'next_mask': torch.tensor([done_traj[t][2]]),
                            'next_rnn_hxs': torch.tensor([0]),
                            })
                else:
                    raise ValueError(f"Invalid HER strategy {self.args.her_strat}")
