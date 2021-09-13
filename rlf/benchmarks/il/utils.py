import torch


def trim_episodes_trans(trajs, orig_trajs):
    """
    Stops the episodes on the first episode the goal is found.
    """
    if "ep_found_goal" not in trajs:
        return trajs
    done = trajs["done"].float()
    obs = trajs["obs"].float()
    next_obs = trajs["next_obs"].float()
    actions = trajs["actions"].float()

    found_goal = trajs["ep_found_goal"].float()

    real_obs = []
    real_done = []
    real_next_obs = []
    real_actions = []
    real_found_goal = []

    num_samples = done.shape[0]
    start_j = 0
    j = 0
    while j < num_samples:
        if found_goal[j] == 1.0:
            real_obs.extend(obs[start_j : j + 1])
            real_done.extend(done[start_j : j + 1])
            real_next_obs.extend(next_obs[start_j : j + 1])
            real_actions.extend(actions[start_j : j + 1])
            real_found_goal.extend(found_goal[start_j : j + 1])

            # Force the episode to end now.
            real_done[-1] = torch.ones(done[-1].shape)

            # Move to where this episode ends
            while j < num_samples and not done[j]:
                j += 1
            start_j = j + 1

        if j < num_samples and done[j]:
            start_j = j + 1

        j += 1

    trajs["done"] = torch.stack(real_done)
    trajs["obs"] = torch.stack(real_obs)
    trajs["next_obs"] = torch.stack(real_next_obs)
    trajs["actions"] = torch.stack(real_actions)
    trajs["ep_found_goal"] = torch.stack(real_found_goal)
    return trajs
