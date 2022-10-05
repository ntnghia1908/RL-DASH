import argparse
import shutil
from evaluators import *
from utils import *
import parse_args_file


def main():
    if not os.path.isdir("results"):
        os.mkdir("results")
    else:
        shutil.rmtree("results")
        os.mkdir("results")

    parser = parse_args_file.parse()
    args = parser.parse_args()
    if args.use_wandb:
        os.environ["WANDB_MODE"] = "online"
        wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, sync_tensorboard=True
        )
        wandb.run.name = args.wandb_runname
        wandb.config.update(args)
    else:
        os.environ["WANDB_MODE"] = "offline"

    seed = args.seed
    # for arg in vars(args):
    #     print("--", end="")
    #     print(arg, getattr(args, arg), end=" \\\n")

    if args.act_func == "tanh":
        activation_fn = torch.nn.modules.activation.Tanh
    elif args.act_func == "relu":
        activation_fn = torch.nn.modules.activation.ReLU
    else:
        raise NotImplementedError("Agent network activation function is not supported.")

    on_policy_kwargs = dict(
        features_extractor_class=PensieveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=[
            dict(
                pi=[args.policy_net_arch_units] * args.policy_net_arch_layers,
                vf=[args.value_net_arch_units] * args.value_net_arch_layers,
            )
        ],
        activation_fn=activation_fn,
    )

    off_policy_kwargs = dict(
        features_extractor_class=PensieveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=[args.value_net_arch_units] * args.value_net_arch_layers,
        activation_fn=activation_fn,
    )

    DQN_params = dict(
        policy_kwargs=off_policy_kwargs,
        learning_rate=args.dqn_lr,
        buffer_size=args.dqn_buffer_size,
        learning_starts=args.dqn_learning_starts,
        batch_size=args.dqn_batch_size,
        tau=args.dqn_tau,
        gamma=args.dqn_gamma,
        train_freq=args.dqn_train_freq,
        gradient_steps=args.dqn_grad_steps,
        target_update_interval=args.dqn_target_update_interval,
        exploration_fraction=args.dqn_exploration_fraction,
        exploration_initial_eps=args.dqn_exploration_initial_eps,
        exploration_final_eps=args.dqn_exploration_final_eps,
        max_grad_norm=args.dqn_max_grad_norm,
    )

    A2C_params = dict(
        policy_kwargs=on_policy_kwargs,
        learning_rate=args.a2c_lr,
        n_steps=args.a2c_n_steps,
        gamma=args.a2c_gamma,
        gae_lambda=args.a2c_gae_lambda,
        ent_coef=args.a2c_ent_coef,
        vf_coef=args.a2c_vf_coef,
        max_grad_norm=args.a2c_max_grad_norm,
        use_rms_prop=args.a2c_rmsprop,
        normalize_advantage=args.a2c_norm_adv,
    )

    PPO_params = dict(
        policy_kwargs=on_policy_kwargs,
        learning_rate=args.ppo_lr,
        n_steps=args.ppo_n_steps_coef * args.ppo_batch_size,
        batch_size=args.ppo_batch_size,
        n_epochs=args.ppo_n_epochs,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_gae_lambda,
        clip_range=args.ppo_clip_range,
        clip_range_vf=args.ppo_clip_range_vf,
        ent_coef=args.ppo_ent_coef,
        vf_coef=args.ppo_vf_coef,
        max_grad_norm=args.ppo_max_grad_norm,
    )

    fcc_train = np.asarray(pickle.load(open("./bw/fcc_train100kb.pickle", "rb")))
    fcc_train = np.repeat(fcc_train, 10, axis=1)
    lte_train = np.asarray(pickle.load(open("./bw/LTE_train100kb.pickle", "rb")))
    train = np.concatenate((fcc_train, lte_train), axis=0)

    fcc_test = np.asarray(pickle.load(open("./bw/fcc_test100kb.pickle", "rb")))
    fcc_test = np.repeat(fcc_test, 10, axis=1)
    lte_test = np.asarray(pickle.load(open("./bw/LTE_test100kb.pickle", "rb")))

    if "DQN" in args.algorithms:
        print("\n\n***EVALUATING: DQN***\n\n")
        dqn_eval = DQNEvaluator(
            EVAL_EPS=args.eval_eps,
            seed=args.seed,
            log_qoe=args.log_qoe,
            bitrate_list=train,
        )
        dqn_eval.evaluate("./results" + str(seed) + "DQN/DQN", params=DQN_params)
        dqn_eval.test(
            "./results" + str(seed) + "DQN/DQN",
            bitrate_list_test_fcc=fcc_test,
            bitrate_list_test_lte=lte_test,
        )
        # result_plotter("results/testDQN.monitor.csv", "DQN")

    if "A2C" in args.algorithms:
        print("\n\n***EVALUATING: A2C***\n\n")
        a2c_eval = A2CEvaluator(
            EVAL_EPS=args.eval_eps,
            seed=args.seed,
            log_qoe=args.log_qoe,
            bitrate_list=train,
            n_envs=args.n_envs,
        )
        a2c_eval.evaluate(
            "./results" + args.dataset + str(seed) + "A2C/A2C", params=A2C_params
        )
        a2c_eval.test(
            "./results" + args.dataset + str(seed) + "A2C/A2C",
            bitrate_list_test_fcc=fcc_test,
            bitrate_list_test_lte=lte_test,
        )

    if "PPO" in args.algorithms:
        print("\n\n***EVALUATING: PPO***\n\n")
        ppo_eval = PPOEvaluator(
            EVAL_EPS=args.eval_eps,
            seed=args.seed,
            log_qoe=args.log_qoe,
            bitrate_list=train,
            n_envs=args.n_envs,
        )
        ppo_eval.evaluate(
            "./results" + args.dataset + str(seed) + "PPO/PPO", params=PPO_params
        )
        ppo_eval.test(
            "./results" + args.dataset + str(seed) + "PPO/PPO",
            bitrate_list_test_fcc=fcc_test,
            bitrate_list_test_lte=lte_test,
        )

    if "Random" in args.other_algorithms:
        print("\n\n***EVALUATING: Random***\n\n")
        random_eval = RandomEvaluator(
            EVAL_EPS=args.eval_eps,
            seed=args.seed,
            log_qoe=args.log_qoe,
            bitrate_list=fcc_test if args.dataset == "FCC" else lte_test,
        )
        random_eval.evaluate("./results/Random.csv")

    if "Smooth" in args.other_algorithms:
        print("\n\n***EVALUATING: Smooth***\n\n")
        smooth_eval = SmoothEvaluator(
            EVAL_EPS=args.eval_eps,
            log_qoe=args.log_qoe,
            bitrate_list=fcc_test if args.dataset == "FCC" else lte_test,
        )
        smooth_eval.evaluate("./results/Smooth.csv")

    if "Constant" in args.other_algorithms:
        print("\n\n***EVALUATING: Constant***\n\n")
        const_eval = ConstantEvaluator(
            EVAL_EPS=args.eval_eps,
            log_qoe=args.log_qoe,
            bitrate_list=fcc_test if args.dataset == "FCC" else lte_test,
        )
        const_eval.evaluate("./results/Constant.csv")

    if "Bola" in args.other_algorithms:
        print("\n\n***EVALUATING: Bola***\n\n")
        bola_evaluate = BolaEvaluator(
            EVAL_EPS=args.eval_eps,
            log_qoe=args.log_qoe,
            bitrate_list=fcc_test if args.dataset == "FCC" else lte_test,
        )
        bola_evaluate.evaluate("./results/Bola.csv")

    if "Offline" in args.other_algorithms:
        print("\n\n***EVALUATING: Offline***\n\n")
        bitrate_list = fcc_test if args.dataset == "FCC" else lte_test
        offline_evaluate = OfflineEvaluator(
            EVAL_EPS=args.eval_eps, log_qoe=args.log_qoe, bitrate_list=bitrate_list
        )
        offline_evaluate.evaluate("./results/Offline.csv")


if __name__ == "__main__":
    main()
    exit()
