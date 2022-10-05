# Evaluating RL algorithms on DASH singlepath

## prerequisite
- Tested on Python 3.8 and 3.9
- PyTorch 1.9.0 must be installed beforehand: [docs](https://pytorch.org/get-started/locally/). CUDA version is not required.

## Usage
Clone the repo, cd to the project folder and run: `pip install -r requirements.txt`

```
usage: main.py [-h] [--eval_eps EVAL_EPS] [--log_qoe LOG_QOE] [--dataset {LTE,FCC}] [--seed SEED]
               [--algorithms {DQN,A2C,PPO,None}] [--other_algorithms OTHER_ALGORITHMS [OTHER_ALGORITHMS ...]]
               [--features_dim FEATURES_DIM] [--policy_net_arch_units POLICY_NET_ARCH_UNITS]
               [--policy_net_arch_layers POLICY_NET_ARCH_LAYERS] [--value_net_arch_units VALUE_NET_ARCH_UNITS]
               [--value_net_arch_layers VALUE_NET_ARCH_LAYERS] [--act_func {tanh,relu}] [--n_envs N_ENVS]
               [--use_wandb USE_WANDB] [--wandb_key WANDB_KEY] [--wandb_entity WANDB_ENTITY]
               [--wandb_project WANDB_PROJECT] [--wandb_runname WANDB_RUNNAME] [--dqn_lr DQN_LR]
               [--dqn_buffer_size DQN_BUFFER_SIZE] [--dqn_learning_starts DQN_LEARNING_STARTS]
               [--dqn_batch_size DQN_BATCH_SIZE] [--dqn_tau DQN_TAU] [--dqn_gamma DQN_GAMMA]
               [--dqn_train_freq DQN_TRAIN_FREQ] [--dqn_grad_steps DQN_GRAD_STEPS]
               [--dqn_target_update_interval DQN_TARGET_UPDATE_INTERVAL]
               [--dqn_exploration_fraction DQN_EXPLORATION_FRACTION]
               [--dqn_exploration_initial_eps DQN_EXPLORATION_INITIAL_EPS]
               [--dqn_exploration_final_eps DQN_EXPLORATION_FINAL_EPS] [--dqn_max_grad_norm DQN_MAX_GRAD_NORM]
               [--a2c_lr A2C_LR] [--a2c_n_steps A2C_N_STEPS] [--a2c_gamma A2C_GAMMA]
               [--a2c_gae_lambda A2C_GAE_LAMBDA] [--a2c_ent_coef A2C_ENT_COEF] [--a2c_vf_coef A2C_VF_COEF]
               [--a2c_max_grad_norm A2C_MAX_GRAD_NORM] [--a2c_rmsprop A2C_RMSPROP] [--a2c_norm_adv A2C_NORM_ADV]
               [--ppo_lr PPO_LR] [--ppo_n_steps_coef PPO_N_STEPS_COEF] [--ppo_batch_size PPO_BATCH_SIZE]
               [--ppo_n_epochs PPO_N_EPOCHS] [--ppo_gamma PPO_GAMMA] [--ppo_gae_lambda PPO_GAE_LAMBDA]
               [--ppo_clip_range PPO_CLIP_RANGE] [--ppo_clip_range_vf PPO_CLIP_RANGE_VF]
               [--ppo_ent_coef PPO_ENT_COEF] [--ppo_vf_coef PPO_VF_COEF] [--ppo_max_grad_norm PPO_MAX_GRAD_NORM]

DASH singlepath evaluation. Hyperparameters are set to the best.

optional arguments:
  -h, --help            show this help message and exit
  --eval_eps EVAL_EPS   Episodes to evaluate the RL algorithms (default: 15000)
  --log_qoe LOG_QOE     whether to use log QoE (default: True)
  --dataset {LTE,FCC}   dataset to evaluate the algorithm (default: FCC)
  --seed SEED           global random seed (default: 1628488846)
  --algorithms {DQN,A2C,PPO,None}
                        Algorithms to run. Must be DQN, A2C and PPO. (default: PPO)
  --other_algorithms OTHER_ALGORITHMS [OTHER_ALGORITHMS ...]
                        Other algorithms to run, including Random, Constant and Smooth (default: ['Random',
                        'Constant', 'Smooth'])
  --features_dim FEATURES_DIM
                        final embedding vector's shape to feed into the policy and value network. The observation
                        will be extracted to have shape (1, features_dim) before feeding into the networks
                        (default: 256)
  --policy_net_arch_units POLICY_NET_ARCH_UNITS
                        policy network unit. Default is 64 units (default: 64)
  --policy_net_arch_layers POLICY_NET_ARCH_LAYERS
                        policy network layer. Default is 2 layers (default: 2)
  --value_net_arch_units VALUE_NET_ARCH_UNITS
                        value network unit. Default is 128. Note that this is also the Q-Network of DQN. (default:
                        128)
  --value_net_arch_layers VALUE_NET_ARCH_LAYERS
                        value network layer. Default is 2 layers (default: 2)
  --act_func {tanh,relu}
                        activation function of policy and value networks (default: tanh)
  --n_envs N_ENVS       number of parallel envs, use in A2C and PPO (default: 4)
  --use_wandb USE_WANDB
                        whether to use wandb online (default: False)
  --wandb_key WANDB_KEY
  --wandb_entity WANDB_ENTITY
  --wandb_project WANDB_PROJECT
  --wandb_runname WANDB_RUNNAME

DQN parameters:
  --dqn_lr DQN_LR       DQN learning rate (default: 0.00015693677366556495)
  --dqn_buffer_size DQN_BUFFER_SIZE
                        DQN max buffer size (default: 5295)
  --dqn_learning_starts DQN_LEARNING_STARTS
                        DQN min buffer size to start learning (default: 735)
  --dqn_batch_size DQN_BATCH_SIZE
                        DQN batch size (default: 295)
  --dqn_tau DQN_TAU     DQN Polyak update coefficient (default: 1.0)
  --dqn_gamma DQN_GAMMA
                        DQN discount factor (default: 0.99)
  --dqn_train_freq DQN_TRAIN_FREQ
                        DQN update the model every dqn_train_freq steps (default: 86)
  --dqn_grad_steps DQN_GRAD_STEPS
                        DQN gradient step each rollout. -1 means to do as many grad steps as steps done in the
                        rollout. (default: 6)
  --dqn_target_update_interval DQN_TARGET_UPDATE_INTERVAL
                        update the target network every this step. (default: 140)
  --dqn_exploration_fraction DQN_EXPLORATION_FRACTION
                        fraction of the training period over which the exploration rate is reduced (default:
                        0.5647684706168141)
  --dqn_exploration_initial_eps DQN_EXPLORATION_INITIAL_EPS
                        initial exploration rate (default: 1.0)
  --dqn_exploration_final_eps DQN_EXPLORATION_FINAL_EPS
                        final exploration rate (default: 0.05)
  --dqn_max_grad_norm DQN_MAX_GRAD_NORM
                        max value for gradient clipping (default: 10.0)

A2C parameters:
  --a2c_lr A2C_LR       A2C learning rate (default: 0.00022306162295897595)
  --a2c_n_steps A2C_N_STEPS
                        A2C num steps to run for each env each update (default: 315)
  --a2c_gamma A2C_GAMMA
                        A2C discount factor (default: 0.99)
  --a2c_gae_lambda A2C_GAE_LAMBDA
                        A2C GAE lambda (default: 1.0)
  --a2c_ent_coef A2C_ENT_COEF
                        A2C entropy coef. Increase to encourage exploration (default: 1e-09)
  --a2c_vf_coef A2C_VF_COEF
                        A2C value function coef (default: 0.29357444939391175)
  --a2c_max_grad_norm A2C_MAX_GRAD_NORM
                        A2C max value for grad clip (default: 0.5)
  --a2c_rmsprop A2C_RMSPROP
                        whether to use RMSProp or Adam as optimizer (default: True)
  --a2c_norm_adv A2C_NORM_ADV
                        whether to use normalized advantage (default: True)

PPO parameters:
  --ppo_lr PPO_LR       PPO learning rate (default: 0.00022678923030566394)
  --ppo_n_steps_coef PPO_N_STEPS_COEF
                        PPO n_steps coef to run for each env per update: n_steps = ppo_n_steps * ppo_batch_size
                        (default: 5)
  --ppo_batch_size PPO_BATCH_SIZE
                        PPO batch size (default: 466)
  --ppo_n_epochs PPO_N_EPOCHS
                        PPO num epochs to run each update (default: 10)
  --ppo_gamma PPO_GAMMA
                        PPO reward discount factor (default: 0.99)
  --ppo_gae_lambda PPO_GAE_LAMBDA
                        PPO GAE lambda coef (default: 0.95)
  --ppo_clip_range PPO_CLIP_RANGE
                        PPO clip range (default: 0.3)
  --ppo_clip_range_vf PPO_CLIP_RANGE_VF
                        PPO Value function clipping parameter (default: None)
  --ppo_ent_coef PPO_ENT_COEF
                        PPO entropy coef (default: 1e-05)
  --ppo_vf_coef PPO_VF_COEF
                        PPO value function coef (default: 0.3784628889100008)
  --ppo_max_grad_norm PPO_MAX_GRAD_NORM
                        PPO Gradient clipping value (default: 0.5)


```