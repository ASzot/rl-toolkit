# RL Toolkit (RLT)

Codebase I use to to implement RL algorithms.

## Algorithms
- On policy:
  - REINFORCE
  - Actor Critic (A2C)
  - Proximal Policy Optimization (PPO)
- Off policy:
  - Deep Q-Networks (DQN)
  - Deep Deterministic Policy Gradients (DDPG)
  - Soft Actor Critic (SAC) 
  - Hindsight Experience Replay (HER)
- Imitation learning:
  - Behavioral Cloning (BC)
  - Generative Adversarial Imitation Learning (GAIL)
  - Adversarial Inverse Reinforcement Learning (AIRL)
- Imitation learning from observation: 
  - Generative Adversarial Imitation from Observation (GAIfO)
  - Behavioral Cloning from Observations (BCO)
- Hierarchical RL: 
  - Option Critic

See learning curves for these algorithms [below](https://github.com/ASzot/rl-toolkit#benchmarks)

## Installation
Requires Python 3.7 or higher. With conda: 

- Clone the repo
- `conda create -n rlf python=3.7`
- `source activate rlf`
- `pip install -r requirements.txt`. 
- `pip install -e .`

If you want to install MuJoCo as well: `mujoco-py==2.0.2.5` 

## More Topics
* `rlf/exp_mgr/README.md` for how to use the automatic job creation and scaling script along with all the plotting utilities.
* `rlf/envs/README.md` for integrating custom environments that take as input command line arguments. 
* `rlf/test/README.md` for details about the unit tests. 

## Benchmarks
* IL Benchmark: `python -m rlf.benchmarks.il`

## Experiment Runner
Easily run templated commands. Start by defining a `.cmd` file. 
- Send to new tmux pane. 
- Easily run and manage long complicated commands. 
- Add additional arguments to specified command. 
- Specify which GPU to use via a flag. 
- Choose to log to W&B. 

## Ray
Install with `pip install ray` and `pip install "ray[tune]"`. To run a job with
Ray specify `--ray` and specify your hyperparam search for Ray tune using
Python syntax in the command line argument with `--ray-config "{'lr':
tune.uniform(0.01, 0.001)}"`. You can specify additional settings such as
`--ray-cpus`, `--ray-gpus`, `--ray-nsamples`. `--ray-debug` runs Ray in serial
mode. When using Ray you cannot reference global variables from anywhere in
your RunSettings.

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
