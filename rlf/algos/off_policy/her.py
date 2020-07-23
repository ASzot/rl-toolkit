from rlf.storage.transition_storage import TransitionStorage
import torch
import rlf.rl.utils as rutils

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
    """

    def _on_traj_done(self, done_trajs):
        for done_traj in done_trajs:
            # augment the trajectory HER style.
            final_goal = done_traj[-1][0]['achieved_goal']

            for i in range(len(done_traj)):
                state = done_traj[i][0]
                state['desired_goal'] = final_goal

                # Is this the final frame?
                if i < len(done_traj) - 1:
                    next_state = done_traj[i+1][0]
                else:
                    # Yes, this is the final frame, there is no next state.
                    # Just an invalid state
                    next_state = state

                if i == 0:
                    mask = 1.0
                else:
                    mask = done_traj[i-1][2]

                self._push_transition({
                        'action': done_traj[i][1],
                        'state': state,
                        'mask': torch.tensor([mask]),
                        'rnn_hxs': torch.tensor([0]),
                        'reward': torch.tensor([done_traj[i][4]]),
                        'next_state': next_state,
                        'next_mask': torch.tensor([done_traj[i][2]]),
                        'next_rnn_hxs': torch.tensor([0]),
                        })
