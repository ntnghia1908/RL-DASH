# How to set up Hyperparam tuning via Wandb sweep
Sweep is wandb's hyperparam tuning tool. A sweep configuration is defined in a `.yaml` file. Sample `yaml` files are given.
A sweep runs the program through CLI, so argparse must be used.

Please see also: [Sweep Docs](https://docs.wandb.ai/guides/sweeps)

A `yaml` file structure is as follows:
```yaml
program: main.py
method: bayes
metric:
  goal: maximize
  name: test_rew_mean
parameters:
  ppo_lr:
    distribution: uniform
    min: 0.00001
    max: 0.001
  ppo_n_steps_coef:
    values: [1, 2, 3, 4, 5]
  ppo_batch_size:
    distribution: int_uniform
    min: 59
    max: 590
  ppo_n_epochs:
    values: [10, 20, 30]
  ppo_gamma:
    values: [0.99, 1.0]
  ppo_gae_lambda:
    values: [0.9, 0.95]
  ppo_clip_range:
    values: [0.2, 0.3]
  ppo_ent_coef:
    values: [0.0, 0.00001, 0.00000001]
  ppo_vf_coef:
    distribution: uniform
    min: 0.2
    max: 0.5

command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--algorithms"
  - "PPO"
  - "--other_algorithms"
  - ""
  - "--dataset"
  - "FCC"
  - "--n_envs"
  - 4
  - "--use_wandb"
  - "true"
  - ${args}
```

Explananation of the file structures:
- `program: main.py`: this is the main file to run. Wandb runs it as follows: `python main.py`
- `method: bayes`: the hyperparameter tuning strategy. 
- `metric, goal` and `name`: defines which metrics to find the optimal value. Will raise an error of the metric is *not* defined.
- `parameters`: are what wandb will pass through CLI to run and tune the program. For example, it runs as: `python main.py --ppo_lr some_value ...`
- `command`: **DO NOT MODIFY THESE**, except the parts between `${program}` and `{$args}`, like follows:
```yaml
- ${program}
- "--algorithms"
- "PPO"
- "--other_algorithms"
- ""
- "--features_dim"
- 256
- "--act_func"
- "tanh"
- "--dataset"
- "FCC"
- "--n_envs"
- 4
- "--use_wandb"
- "true"
- ${args}
```

The parts between `${program}` and `{$args}` are the **default** arguments to pass through CLI. This config is the same across all runs in a sweep.
For some reason, **it does not work for `nargs="+"` argparse**.

## Initialize and run a sweep
First, run: `wandb login "your_security_key"`. **For security reasons, please create a wandb account yourself, and use your own security keys**.

After the `yaml` file is finished, use the command: `wandb sweep your_sweep_yaml.yaml`. It will return an ID. Then initialize the runs as follows:
`wandb agent --count N "returned_run_id"`, where `N` is the number of times you want it to run.

