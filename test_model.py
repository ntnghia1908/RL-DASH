from singlepath_gym import SinglepathEnvGym
from stable_baselines3.common.evaluation import evaluate_policy
import os
import pickle
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


logger = configure("./test_logger/", ["stdout", "csv"])
EVAL_EPS = 200
info_keywords = ("reward_quality_norm", "reward_smooth_norm", "reward_rebuffering_norm")

for dataset in ["FCC"]:
    # fcc_train = np.asarray(pickle.load(open('./bw/fcc_train100kb.pickle', 'rb')))
    # fcc_train = np.repeat(fcc_train, 10, axis=1)
    # lte_train = np.asarray(pickle.load(open('./bw/LTE_train100kb.pickle', 'rb')))
    # train = np.concatenate((fcc_train, lte_train), axis=0)

    if "FCC" in dataset:
        # fcc_train = np.asarray(pickle.load(open('./bw/fcctrain_bw.pickle', 'rb')))
        # train = np.repeat(fcc_train, 10, axis=1)
        fcc_test = np.asarray(pickle.load(open("./bw/fcc_test100kb.pickle", "rb")))
        test = np.repeat(fcc_test, 10, axis=1)
    elif "LTE" in dataset:
        # train = np.asarray(pickle.load(open('./bw/LTE_train.pickle', 'rb')))
        test = np.asarray(pickle.load(open("./bw/LTE_test100kb.pickle", "rb")))
    else:
        assert False

    a2c_rew = []
    dqn_rew = []
    ppo_rew = []

    for cur_dir in os.listdir("final_results/models"):
        for run in os.listdir(os.path.join("final_results/models", cur_dir)):
            for model_path in os.listdir(
                "final_results/models" + "/" + cur_dir + "/" + run
            ):
                env = SinglepathEnvGym(
                    log_qoe=True, bitrate_list=test, replace=True, train=False
                )
                env = Monitor(
                    env,
                    filename="./" + str(run) + str(dataset),
                    info_keywords=info_keywords,
                )
                if "A2C" in cur_dir:
                    model = A2C.load(
                        "final_results/models"
                        + "/"
                        + cur_dir
                        + "/"
                        + run
                        + "/"
                        + model_path
                        + "/DQNEvaluator/model.zip"
                    )
                    list_rew, _ = evaluate_policy(
                        model,
                        env,
                        n_eval_episodes=EVAL_EPS,
                        return_episode_rewards=True,
                    )
                    print(f"Algo: {cur_dir} evaluated on {dataset}")
                    a2c_rew.append(list_rew)
                if "PPO" in cur_dir:
                    model = PPO.load(
                        "final_results/models"
                        + "/"
                        + cur_dir
                        + "/"
                        + run
                        + "/"
                        + model_path
                        + "/DQNEvaluator/model.zip"
                    )
                    list_rew, _ = evaluate_policy(
                        model,
                        env,
                        n_eval_episodes=EVAL_EPS,
                        return_episode_rewards=True,
                    )
                    print(f"Algo: {cur_dir} evaluated on {dataset}")
                    ppo_rew.append(list_rew)
                elif "DQN" in cur_dir:
                    model = DQN.load(
                        "final_results/models"
                        + "/"
                        + cur_dir
                        + "/"
                        + run
                        + "/"
                        + model_path
                        + "/DQNEvaluator/model.zip"
                    )
                    list_rew, _ = evaluate_policy(
                        model,
                        env,
                        n_eval_episodes=EVAL_EPS,
                        return_episode_rewards=True,
                    )
                    print(f"Algo: {cur_dir} evaluated on {dataset}")
                    dqn_rew.append(list_rew)

    a2c = np.array(a2c_rew)
    dqn = np.array(dqn_rew)
    ppo = np.array(ppo_rew)

    np.save("A2C_rew_FCCon" + dataset, a2c)
    np.save("DQN_rew_FCCon" + dataset, dqn)
    np.save("PPO_rew_FCCon" + dataset, ppo)
# print(a2c)
# print(dqn)
# print(ppo)
