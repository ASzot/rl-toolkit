# Reinforcement Learning Toolkit

Code base I use to help with reinforcement learning (RL) research projects.

## Running
Run using
```
python -m rlf.examples.train --alg ALGORITHM_NAME --env-name Pendulum-v1
```
Supported algorithms (to replace `ALGORITHM_NAME` in the above command) 
* Imitation learning (You must also specify the `--traj-load-path` argument for these commands to load the demonstrations. See ["how to specify demonstrations for imitation learning?"](https://github.com/ASzot/rl-toolkit#how-to-specify-demonstrations-for-imitation-learning) for more information.
    * Generative Adversarial Imitation Learning (GAIL): `--alg gail_ppo`
    * Generative Adversarial Imitation Learning from Observations (GAIfO): `--alg gaifo_ppo`
    * Behavioral Cloning (BC): `--alg bc`
    * Behavioral Cloning from Observations (BCO): `--alg bco`
    * Soft-Q Imitation Learning (SQIL): `--alg sqil`
* Reinforcement learning
    * Proximal Policy Optimization (PPO): `--alg ppo`
    * Soft Actor Critic (SAC): `--alg sac`
    * Deep Deterministic Policy Gradients (DDPG): `--alg ddpg`
    * Random Policy: `--alg rnd`

To see the list of all possible command line arguments add `-v`. For example: `python examples/train.py --alg sac --env-name Pendulum-v1 --cuda False -v`. Command line arguments are added by the algorithm or policy. See examples [here](https://github.com/ASzot/rl-toolkit/blob/1edcb1ed12abbf2c8691a1bf8bba56294d1f4c31/rlf/algos/il/gail.py#L301) and [here](https://github.com/ASzot/rl-toolkit/blob/1edcb1ed12abbf2c8691a1bf8bba56294d1f4c31/rlf/args.py#L50).  See learning curves for these algorithms [below](https://github.com/ASzot/rl-toolkit#benchmarks).

## How to use new environments?
* Specify the name of your algorithm using `--env-name`.
* If they are registered through `gym.envs.registration` it will work automatically through `gym.make`.
* See [this page](https://github.com/ASzot/rl-toolkit/tree/master/rlf/envs#readme) for information about specifying custom command line arguments for environments.

## How to specify demonstrations for imitation learning?
See [this comment](https://github.com/ASzot/rl-toolkit/blob/1edcb1ed12abbf2c8691a1bf8bba56294d1f4c31/rlf/il/il_dataset.py#L26) for the demonstration dataset specification. Then load in the demonstration dataset via the `--traj-load-path` argument.


## Installation
Requires Python 3.7 or higher. With conda: 

- Clone the repo
- `conda create -n rlf python=3.7`
- `source activate rlf`
- `pip install -r requirements.txt`
- `pip install -e .`


# Benchmarks
### Hopper-v3

Commit: `570d8c8d024cb86266610e72c5431ef17253c067`
- PPO: `python -m rlf --cmd ppo/hopper --cd 0 --cfg ./tests/config.yaml --seed "31,41,51" --sess-id 0 --cuda False` 

![Hopper-v3](https://github.com/ASzot/rl-toolkit/blob/master/bench_plots/hopper.png)

### HalfCheetah-v3
Commit: `58644db1ac638ba6c8a22e7a01eacfedffd4a49f`
- PPO: `python -m rlf --cmd ppo/halfcheetah --cd 0 --cfg ./tests/config.yaml --seed "31,41,51" --sess-id 0 --cuda False`

![Hopper-v3](https://github.com/ASzot/rl-toolkit/blob/master/bench_plots/halfcheetah.png)

### HalfCheetah-v3 Imitation Learning
Commit: `58644db1ac638ba6c8a22e7a01eacfedffd4a49f`
- BCO: `python -m rlf --cmd bco/halfcheetah --cfg ./tests/config.yaml --seed "31,41,51" --sess-id 0 --cuda False` 
- GAIfO-s: `python -m rlf --cmd gaifo_s/halfcheetah --cfg ./tests/config.yaml --seed "31,41,51" --sess-id 0 --cuda False` 
- GAIfO: `python -m rlf --cmd gaifo/halfcheetah --cfg ./tests/config.yaml --seed "31,41,51" --sess-id 0 --cuda False` 

![Hopper-v3](https://github.com/ASzot/rl-toolkit/blob/master/bench_plots/halfcheetah_il.png)

### Pendulum-v0
Commit: `5c051769088b6582b0b31db9a145738a9ed68565`
- DDPG: `python -m rlf --cmd ddpg/pendulum --cd 0 --cfg ./tests/config.yaml --seed "31,41" --sess-id 0 --cuda False`

![Pendulum-v0](https://github.com/ASzot/rl-toolkit/blob/master/bench_plots/pendulum.png)

### HER
Commit: `95bb3a7d0bf1945e414a0e77de8a749bd79dc554`
- BitFlip: `python -m rlf --cmd her/bit_flip --cfg ./tests/config.yaml --cuda False --sess-id 0`

![HER](https://github.com/ASzot/rl-toolkit/blob/master/bench_plots/her.png)

# Sources
* The SAC code is a clone of https://github.com/denisyarats/pytorch_sac.
  The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
* The PPO and rollout storage code is based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
* The environment preprocessing uses a stripped down version of https://github.com/openai/baselines.
