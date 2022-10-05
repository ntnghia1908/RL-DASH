from numpy.core.fromnumeric import repeat
from singlepath_gym import SinglepathEnvGym
import numpy as np

# from utils import DownloadPath
from stable_baselines3 import PPO, DQN, A2C
from feature_extractor import PensieveFeatureExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
import time
import datetime
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
import itertools
from copy import deepcopy
import os, random, numpy as np, torch
from stable_baselines3.common.env_checker import check_env


def set_global_seed(env, seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env.seed(seed)
    env.action_space.seed(seed)


NUM_STEP_PER_EP = 59
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class BaseEvaluator:
    """
    Base evaluator for inheritance
    """

    def __init__(
        self, EVAL_EPS, seed, log_qoe=True, bitrate_list=None, replace=True, n_envs=1
    ):
        self.env = SinglepathEnvGym(
            log_qoe=log_qoe, bitrate_list=bitrate_list, replace=replace, train=True
        )
        print("CHECKING ENVIRONMENT")
        check_env(self.env)
        print("ENVIRONMENT CHECKING PASSED")
        self.env = SinglepathEnvGym(
            log_qoe=log_qoe, bitrate_list=bitrate_list, replace=replace, train=True
        )
        set_global_seed(self.env, seed)
        self.env_kwargs = dict(
            log_qoe=log_qoe, bitrate_list=bitrate_list, replace=replace, train=True
        )
        self.n_envs = n_envs
        self.M_IN_K = self.env.M_IN_K
        self.HISTORY_SIZE = self.env.HISTORY_SIZE
        self.QUALITY_SPACE = self.env.QUALITY_SPACE
        self.CHUNK_TIL_VIDEO_END = self.env.CHUNK_TIL_VIDEO_END
        self.SCALE = 0.9
        self.VIDEO_CHUNK_LEN = self.env.VIDEO_CHUNK_LEN
        self.EVAL_EPS = EVAL_EPS
        self.n_envs = n_envs
        self.print_template = (
            "{0:^20}|{1:^20}|{2:^20}|{3:^20}|{4:^20}|{5:^20}|{6:^20}\n"
        )
        self.write_template = "{},{},{},{},{},{},{}\n"

    def predict(self, *args):
        pass

    def evaluate(self, *args):
        pass


class ConstantEvaluator(BaseEvaluator):
    """
    An dummy evaluator using constant prediction.

    Always return the max quality.
    """

    def predict(self):
        # Always return max quality
        return 4

    def evaluate(self, file_name):
        self.env.train = False
        with open(file_name, "a+") as file:
            file.write(
                self.write_template.format(
                    "ep",
                    "utility",
                    "switch_penalty",
                    "rebuffer_penalty",
                    "no_switch",
                    "rebuffer_time",
                    "total_reward",
                )
            )

        print(
            self.print_template.format(
                "ep",
                "utility",
                "switch_penalty",
                "rebuffer_penalty",
                "no_switch",
                "rebuffer_time",
                "total_reward",
            ),
            end="",
        )
        for eps in range(200):
            self.env.reset()
            while not self.env.end_of_video:
                predicted_action = self.predict()
                state, reward, done, info = self.env.step(predicted_action)
            print(
                self.print_template.format(
                    *np.around(
                        [
                            eps + 1,
                            self.env.reward_quality_norm,
                            self.env.reward_smooth_norm,
                            self.env.reward_rebuffering_norm,
                            self.env.num_switch,
                            self.env.rebuffer_time,
                            self.env.total_reward,
                        ],
                        5,
                    )
                ),
                end="",
            )

            with open(file_name, "a+") as file:
                file.write(
                    self.write_template.format(
                        *np.around(
                            [
                                eps + 1,
                                self.env.reward_quality_norm,
                                self.env.reward_smooth_norm,
                                self.env.reward_rebuffering_norm,
                                self.env.num_switch,
                                self.env.rebuffer_time,
                                self.env.total_reward,
                            ],
                            5,
                        )
                    )
                )


class SmoothEvaluator(BaseEvaluator):
    """
    Smooth Throughput evaluator.
    Use mean (or harmonic mean) of window_size (typically 3) previous network speed to predict the next quality,
        such that it does not exceed the bitrates of the next segment.
    """

    def predict(
        self, video_list, prev_network_speed, rule="harmonic_mean", window_size=3
    ):
        """
        Predict the next qualities based on previous network speed.
        :param video_list: np.array of shape (7, CHUNK_TIL_VIDE_END), row is the quality level and column is its sizes.
            It is self.env.video_list
        :param prev_network_speed: np.array of shape (HISTORY_SIZE,), previous network speed of a path
        :param rule: string, either mean or harmonic_mean, indicates which rule to calculate quality
        :param window_size: int, default 3, how many previous network speed values to use.
        :return:
        """
        down_id = self.env.pick_next_segment()
        segment_bitrates = video_list[:, down_id] * 8 / self.VIDEO_CHUNK_LEN
        prev_network_speed = prev_network_speed[:window_size].copy()
        prev_network_speed = prev_network_speed[
            prev_network_speed > 0
        ].copy()  # Eliminate zeros (values that are unfilled)
        if len(prev_network_speed) == 0:
            return 0
        if rule == "mean":
            predicted_quality = sum(prev_network_speed) / len(prev_network_speed)
        elif rule == "harmonic_mean":
            predicted_quality = len(prev_network_speed) / np.sum(1 / prev_network_speed)
        else:
            raise AssertionError("Rule is not mean or harmonic_mean")

        picked_quality = 0
        if (
            segment_bitrates[picked_quality]
            >= predicted_quality * self.SCALE * self.M_IN_K ** 2
        ):
            return picked_quality
        while (
            segment_bitrates[picked_quality]
            < predicted_quality * self.SCALE * self.M_IN_K ** 2
        ):
            picked_quality += 1
            if picked_quality == self.QUALITY_SPACE:
                break
        return picked_quality - 1

    def evaluate(self, file_name):
        self.env.train = False
        print(
            self.print_template.format(
                "ep",
                "utility",
                "switch_penalty",
                "rebuffer_penalty",
                "no_switch",
                "rebuffer_time",
                "total_reward",
            ),
            end="",
        )

        with open(file_name, "a+") as file:
            file.write(
                self.write_template.format(
                    "ep",
                    "utility",
                    "switch_penalty",
                    "rebuffer_penalty",
                    "no_switch",
                    "rebuffer_time",
                    "total_reward",
                )
            )

        for eps in range(200):
            # cur_path = DownloadPath.PATH1
            state = self.env.reset()
            while not self.env.end_of_video:
                predicted_action = self.predict(
                    self.env.video_list, state["network_speed"]
                )
                state, reward, done, info = self.env.step(predicted_action)
                # cur_path = info["cur_down_path"]

            print(
                self.print_template.format(
                    *np.around(
                        [
                            eps + 1,
                            self.env.reward_quality_norm,
                            self.env.reward_smooth_norm,
                            self.env.reward_rebuffering_norm,
                            self.env.num_switch,
                            self.env.rebuffer_time,
                            self.env.total_reward,
                        ],
                        5,
                    )
                ),
                end="",
            )
            with open(file_name, "a+") as file:
                file.write(
                    self.write_template.format(
                        *np.around(
                            [
                                eps + 1,
                                self.env.reward_quality_norm,
                                self.env.reward_smooth_norm,
                                self.env.reward_rebuffering_norm,
                                self.env.num_switch,
                                self.env.rebuffer_time,
                                self.env.total_reward,
                            ],
                            5,
                        )
                    )
                )


class RandomEvaluator(BaseEvaluator):
    def predict(self):
        return np.random.choice(np.arange(7))

    def evaluate(self, file_name):
        self.env.train = False
        with open(file_name, "a+") as file:
            file.write(
                self.write_template.format(
                    "ep",
                    "utility",
                    "switch_penalty",
                    "rebuffer_penalty",
                    "no_switch",
                    "rebuffer_time",
                    "total_reward",
                )
            )

        print(
            self.print_template.format(
                "ep",
                "utility",
                "switch_penalty",
                "rebuffer_penalty",
                "no_switch",
                "rebuffer_time",
                "total_reward",
            ),
            end="",
        )
        for eps in range(200):
            self.env.reset()
            while not self.env.end_of_video:
                predicted_action = self.predict()
                state, reward, done, info = self.env.step(predicted_action)

            print(
                self.print_template.format(
                    *np.around(
                        [
                            eps + 1,
                            self.env.reward_quality_norm,
                            self.env.reward_smooth_norm,
                            self.env.reward_rebuffering_norm,
                            self.env.num_switch,
                            self.env.rebuffer_time,
                            self.env.total_reward,
                        ],
                        5,
                    )
                ),
                end="",
            )

            with open(file_name, "a+") as file:
                file.write(
                    self.write_template.format(
                        *np.around(
                            [
                                eps + 1,
                                self.env.reward_quality_norm,
                                self.env.reward_smooth_norm,
                                self.env.reward_rebuffering_norm,
                                self.env.num_switch,
                                self.env.rebuffer_time,
                                self.env.total_reward,
                            ],
                            5,
                        )
                    )
                )


class BolaEvaluator(BaseEvaluator):
    def predict(
        self, playtime_from_begin, playtime_to_end, prev_quality, cur_buffer, prev_bw
    ):
        """
        Implement BOLA Algorithm

        Args:
            Q(t_k): the buffer level at the start of the slot k (in seconds).\n
            Q_max: buffer therhhold (in seconds).\n
            Q^D_max: dynamic buffer level. (in seconds).\n
            Q: current buffer level (in second).\n
            S_m: the size of any segment encoded at bitrate index m  (in bits).\n
            v_m: ln(S_m/S_1) utility function.\n
            V>0: 0.93 to allow a tradeoff between the buffer size and the performance objectives.\n
            V_D: Dynamic V which corresponds to a dynamic buffer size Q^D_max.\n
            m*: The index that maximizes the ratio among all m for which this ratio is positive.\n
            m*[n]: Size of segment n at bitrate index m*.\n
            m*[n-1]: Size of segment n-1 at bitrate index m*.\n
            p: video segment (in second)


        :param (,7) next_seg: n segment sizes need choice one for download (in bps)
        :param float playtime_from_begin:
        :param float playtime_to_end:
        :param float prev_quality: bandwidth measured when downloading segment n-1
        :param float cur_buffer: buffer level at the time start to download segment n
        :param float prev_bw: estimated bandwidth (in bps) on the considering path
        :param (,7) S_m: the size of any segment encoded at bitrate index m  (in bits)
        :return: the quality index for download
        """

        p = self.VIDEO_CHUNK_LEN
        Q_MAX = self.env.BUFFER_THRESHOLD / p
        S_m = np.array(self.env.VIDEO_BIT_RATE) * 4.0 * self.M_IN_K

        v = np.log(S_m / S_m[0])
        gamma = 5.0 / p  # CHANGE from 5.0/p

        t = min(playtime_from_begin, playtime_to_end)
        t_prime = max(t / 2.0, 3 * p)
        Q_D_max = min(Q_MAX, t_prime / p)
        V_D = (Q_D_max - 1) / (v[-1] + gamma * p)  # v[-1] is v_M

        m_star = 0
        score = None
        for q in range(len(S_m)):
            s = (V_D * (v[q] + p * gamma) - cur_buffer / p) / S_m[q]
            if score is None or s > score:
                m_star = q
                score = s

        if m_star > prev_quality:
            r = prev_bw * 10 ** 6  # Calculate in bits
            m_prime = np.where(S_m / p <= max(r, S_m[0] / p))[0][-1]
            if m_prime >= m_star:
                m_prime = m_star
            elif m_prime < prev_quality:
                m_prime = prev_quality

            else:
                m_prime = m_prime + 1
            m_star = m_prime
        return m_star

    def evaluate(self, file_name):
        self.env.train = False
        print(
            self.print_template.format(
                "ep",
                "utility",
                "switch_penalty",
                "rebuffer_penalty",
                "no_switch",
                "rebuffer_time",
                "total_reward",
            ),
            end="",
        )

        with open(file_name, "a+") as file:
            file.write(
                self.write_template.format(
                    "ep",
                    "utility",
                    "switch_penalty",
                    "rebuffer_penalty",
                    "no_switch",
                    "rebuffer_time",
                    "total_reward",
                )
            )

        for eps in range(200):
            # cur_path = DownloadPath.PATH1
            state = self.env.reset()
            playtime_from_begin = 0
            prev_network_speed = 0
            while not self.env.end_of_video:
                down_id = self.env.pick_next_segment()
                playtime_to_end = (
                    self.CHUNK_TIL_VIDEO_END - self.env.play_id * self.VIDEO_CHUNK_LEN
                )
                prev_quality = self.env.download_segment[down_id - 1]
                cur_buffer = self.env.buffer_size_trace
                predicted_action = self.predict(
                    playtime_from_begin,
                    playtime_to_end,
                    prev_quality,
                    cur_buffer,
                    prev_network_speed,
                )
                state, reward, done, info = self.env.step(predicted_action)

                prev_network_speed = state["network_speed"][0]
                playtime_from_begin = (
                    self.env.play_id * self.VIDEO_CHUNK_LEN + self.env.rebuffer_time
                )

            print(
                self.print_template.format(
                    *np.around(
                        [
                            eps + 1,
                            self.env.reward_quality_norm,
                            self.env.reward_smooth_norm,
                            self.env.reward_rebuffering_norm,
                            self.env.num_switch,
                            self.env.rebuffer_time,
                            self.env.total_reward,
                        ],
                        5,
                    )
                ),
                end="",
            )
            with open(file_name, "a+") as file:
                file.write(
                    self.write_template.format(
                        *np.around(
                            [
                                eps + 1,
                                self.env.reward_quality_norm,
                                self.env.reward_smooth_norm,
                                self.env.reward_rebuffering_norm,
                                self.env.num_switch,
                                self.env.rebuffer_time,
                                self.env.total_reward,
                            ],
                            5,
                        )
                    )
                )


class MPCEvaluator(BaseEvaluator):
    MPC_FUTURE_CHUNK_COUNT = 5
    CHUNK_COMBO_OPTIONS = []
    # TOTAL_VIDEO_CHUNKS = self.CHUNK_TIL_VIDEO_END

    def predict(self, video_list, last_quality, buffer_size, future_bandwidth):
        down_id = self.env.pick_next_segment()

        for combo in itertools.product(np.arange(self.env.QUALITY_SPACE), repeat=5):
            self.CHUNK_COMBO_OPTIONS.append(combo)

        # future chunks length (try 4 if that many remaining)
        future_chunk_length = self.MPC_FUTURE_CHUNK_COUNT
        if self.CHUNK_TIL_VIDEO_END - down_id < 5:
            future_chunk_length = self.CHUNK_TIL_VIDEO_END - down_id

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size
        # start = time.time()
        for full_combo in self.CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = (
                    down_id + position
                )  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                segment_bitrates = (
                    self.env.video_list[:, index] * 8 / self.VIDEO_CHUNK_LEN
                )
                download_time = (
                    segment_bitrates[chunk_quality] / 1000000.0
                ) / future_bandwidth  # this is MB/MB/s --> seconds
                if curr_buffer < download_time:
                    curr_rebuffer_time += download_time - curr_buffer
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += self.VIDEO_CHUNK_LEN
                bitrate_sum += self.env.UTILITY_SCORE[chunk_quality]
                smoothness_diffs += abs(
                    self.env.UTILITY_SCORE[chunk_quality]
                    - self.env.UTILITY_SCORE[last_quality]
                )

                last_quality = chunk_quality

            # compute reward for this combination (one reward per 5-chunk combo)
            combo_reward = (
                bitrate_sum / 100
                - self.env.REBUF_PENALTY * curr_rebuffer_time / 100
                - self.SMOOTH_PENALTY * smoothness_diffs
            )
            # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)

            if combo_reward >= max_reward:
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = combo_reward
                # send data to html side (first chunk of best combo)
                chunk_quality = (
                    0  # no combo had reward better than -1000000 (ERROR) so send 0
                )
                if best_combo != ():  # some combo was good
                    chunk_quality = best_combo[0]

        return chunk_quality

    def evaluate(self, file_name):
        self.env.train = False
        print(
            self.print_template.format(
                "ep",
                "utility",
                "switch_penalty",
                "rebuffer_penalty",
                "no_switch",
                "rebuffer_time",
                "total_reward",
            ),
            end="",
        )

        with open(file_name, "a+") as file:
            file.write(
                self.write_template.format(
                    "ep",
                    "utility",
                    "switch_penalty",
                    "rebuffer_penalty",
                    "no_switch",
                    "rebuffer_time",
                    "total_reward",
                )
            )

        for eps in range(200):
            state = self.env.reset()
            # playtime_from_begin = 0
            # prev_network_speed = 0
            curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
            past_errors = []
            past_bandwidth_ests = []

            while not self.env.end_of_video:
                down_id = self.env.pick_next_segment()
                if len(state["network_speed"]) < 5:
                    past_bandwidths = state["network_speed"]
                else:
                    past_bandwidths = state["network_speed"][-5:]

                if len(past_bandwidth_ests) > 0:
                    curr_error = abs(
                        past_bandwidth_ests[-1] - past_bandwidths[-1]
                    ) / float(past_bandwidths[-1])
                past_errors.append(curr_error)

                # pick bitrate according to MPC
                # first get harmonic mean of last window past segment

                bandwidth_sum = 0
                for past_val in past_bandwidths:
                    bandwidth_sum += 1 / float(past_val)
                harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
                # harmonic_bandwidth = len(past_bandwidths) / np.sum(1 / past_bandwidths)

                # future bandwidth prediction
                # divide by 1 + max of last 5 (or up to 5) errors
                max_error = 0
                error_pos = -len(past_bandwidths)
                if len(past_errors) < len(past_bandwidths):
                    error_pos = -len(past_errors)
                max_error = float(max(past_errors[error_pos:]))
                # future_bandwidth = harmonic_bandwidth
                future_bandwidth = harmonic_bandwidth / (
                    1 + max_error
                )  # robustMPC here
                past_bandwidth_ests.append(harmonic_bandwidth)

                last_bit_rate = state["last_down_quality"]
                cur_buffer = self.env.buffer_size_trace
                cur_time = self.env.event[0][0]

                predicted_action = self.predict(
                    self.env.video_list, last_bit_rate, cur_time, cur_buffer
                )

                state, reward, done, info = self.env.step(predicted_action)

            print(
                self.print_template.format(
                    *np.around(
                        [
                            eps + 1,
                            self.env.reward_quality_norm,
                            self.env.reward_smooth_norm,
                            self.env.reward_rebuffering_norm,
                            self.env.num_switch,
                            self.env.rebuffer_time,
                            self.env.total_reward,
                        ],
                        5,
                    )
                ),
                end="",
            )
            with open(file_name, "a+") as file:
                file.write(
                    self.write_template.format(
                        *np.around(
                            [
                                eps + 1,
                                self.env.reward_quality_norm,
                                self.env.reward_smooth_norm,
                                self.env.reward_rebuffering_norm,
                                self.env.num_switch,
                                self.env.rebuffer_time,
                                self.env.total_reward,
                            ],
                            5,
                        )
                    )
                )


class OfflineEvaluator(BaseEvaluator):
    MAX_FUTURE_CHUNK_LENGTH = 3

    def predict(self, env: SinglepathEnvGym) -> int:
        down_id = env.pick_next_segment()
        current_future_chunk_length = min(
            self.MAX_FUTURE_CHUNK_LENGTH, SinglepathEnvGym.CHUNK_TIL_VIDEO_END - down_id
        )

        all_possible_combos = torch.cartesian_prod(
            *torch.arange(SinglepathEnvGym.QUALITY_SPACE).repeat(
                current_future_chunk_length, 1
            )
        )

        combo_rewards = torch.zeros(len(all_possible_combos))

        for i in range(len(all_possible_combos)):
            copy_env = deepcopy(env)

            if len(all_possible_combos[i].shape) == 0:
                _, reward, _, _ = copy_env.step(all_possible_combos[i].item())
                combo_rewards[i] += reward

            else:
                for action in all_possible_combos[i]:
                    _, reward, _, _ = copy_env.step(action.item())
                    combo_rewards[i] += reward

        best_combo_idx = torch.argmax(combo_rewards).item()

        if len(all_possible_combos.shape) == 2:
            best_quality = all_possible_combos[best_combo_idx, 0].item()
        else:
            best_quality = all_possible_combos[best_combo_idx].item()

        return best_quality

    def evaluate(self, file_name):
        self.env.train = False
        print(
            self.print_template.format(
                "ep",
                "utility",
                "switch_penalty",
                "rebuffer_penalty",
                "no_switch",
                "rebuffer_time",
                "total_reward",
            ),
            end="",
        )

        with open(file_name, "a+") as file:
            file.write(
                self.write_template.format(
                    "ep",
                    "utility",
                    "switch_penalty",
                    "rebuffer_penalty",
                    "no_switch",
                    "rebuffer_time",
                    "total_reward",
                )
            )

        for eps in range(200):
            self.env.reset()
            done = False

            while not done:
                picked_action = self.predict(self.env)
                _, _, done, _ = self.env.step(picked_action)

            print(
                self.print_template.format(
                    *np.around(
                        [
                            eps + 1,
                            self.env.reward_quality_norm,
                            self.env.reward_smooth_norm,
                            self.env.reward_rebuffering_norm,
                            self.env.num_switch,
                            self.env.rebuffer_time,
                            self.env.total_reward,
                        ],
                        5,
                    )
                ),
                end="",
            )
            with open(file_name, "a+") as file:
                file.write(
                    self.write_template.format(
                        *np.around(
                            [
                                eps + 1,
                                self.env.reward_quality_norm,
                                self.env.reward_smooth_norm,
                                self.env.reward_rebuffering_norm,
                                self.env.num_switch,
                                self.env.rebuffer_time,
                                self.env.total_reward,
                            ],
                            5,
                        )
                    )
                )


class DQNEvaluator(BaseEvaluator):
    def evaluate(self, file_name, params, save=True, use_wandb=False):
        self.name = "DQN"
        self.logger = configure("./loggerDQN/", ["stdout", "csv", "tensorboard"])
        start_time = time.time()
        # self.env = Monitor(self.env, file_name)
        self.model = DQN(
            "MultiInputPolicy",
            self.env,
            verbose=1,
            **params,
            device=device,
            tensorboard_log=f"runs/{start_time}DQN",
        )
        self.model.set_logger(logger=self.logger)
        self.model.learn(
            total_timesteps=NUM_STEP_PER_EP * self.EVAL_EPS,
            callback=WandbCallback(
                verbose=1, model_save_path="./models/DQNEvaluator" + str(start_time)
            )
            if use_wandb
            else None,
        )
        end_time = time.time()
        seconds_elapsed = end_time - start_time
        print(f"Time elapsed: {str(datetime.timedelta(seconds=seconds_elapsed))}")

    def test(self, file_name, bitrate_list_test_fcc, bitrate_list_test_lte):
        print("Evaluating on FCC...")
        env = SinglepathEnvGym(bitrate_list=bitrate_list_test_fcc, train=False)
        info_keywords = (
            "reward_quality_norm",
            "reward_smooth_norm",
            "reward_rebuffering_norm",
        )
        env = Monitor(
            env, filename="./test_monitorFCC" + self.name, info_keywords=info_keywords
        )
        avg_reward, _ = evaluate_policy(
            self.model, env, n_eval_episodes=200, return_episode_rewards=True
        )
        wandb.log(
            {"test_rew_mean": np.mean(avg_reward), "test_rew_std": np.std(avg_reward)}
        )
        print(f"Algorithm evaluation average reward: {avg_reward}")
        wandb.save("./test_monitorFCC" + self.name + ".monitor.csv")

        print("Evaluating on LTE...")
        env = SinglepathEnvGym(bitrate_list=bitrate_list_test_lte, train=False)
        info_keywords = (
            "reward_quality_norm",
            "reward_smooth_norm",
            "reward_rebuffering_norm",
        )
        env = Monitor(
            env, filename="./test_monitorLTE" + self.name, info_keywords=info_keywords
        )
        avg_reward, _ = evaluate_policy(
            self.model, env, n_eval_episodes=200, return_episode_rewards=True
        )
        wandb.log(
            {
                "test_rew_mean_lte": np.mean(avg_reward),
                "test_rew_std_lte": np.std(avg_reward),
            }
        )
        print(f"Algorithm evaluation average reward: {avg_reward}")
        wandb.save("./test_monitorLTE" + self.name + ".monitor.csv")


class A2CEvaluator(DQNEvaluator):
    def evaluate(self, file_name, params, save=True, use_wandb=False):
        self.name = "A2C"
        self.logger = configure("./loggerA2C/", ["stdout", "csv", "tensorboard"])
        start_time = time.time()
        self.env = make_vec_env(
            SinglepathEnvGym,
            n_envs=self.n_envs,
            monitor_dir=file_name,
            env_kwargs=self.env_kwargs,
        )
        self.model = A2C(
            "MultiInputPolicy",
            self.env,
            verbose=1,
            **params,
            device=device,
            tensorboard_log=f"runs/{time.time()}A2C",
        )
        self.model.set_logger(logger=self.logger)
        self.model.learn(
            total_timesteps=NUM_STEP_PER_EP * self.EVAL_EPS,
            callback=WandbCallback(
                verbose=1, model_save_path="./models/A2CEvaluator" + str(start_time)
            )
            if use_wandb
            else None,
        )
        end_time = time.time()
        seconds_elapsed = end_time - start_time
        print(f"Time elapsed: {str(datetime.timedelta(seconds=seconds_elapsed))}")


class PPOEvaluator(DQNEvaluator):
    def evaluate(self, file_name, params, save=True, use_wandb=False):
        self.name = "PPO"
        self.logger = configure("./loggerPPO/", ["stdout", "csv", "tensorboard"])
        start_time = time.time()
        self.env = make_vec_env(
            SinglepathEnvGym,
            n_envs=self.n_envs,
            monitor_dir=file_name,
            env_kwargs=self.env_kwargs,
        )
        self.model = PPO(
            "MultiInputPolicy",
            self.env,
            verbose=1,
            **params,
            device=device,
            tensorboard_log=f"runs/{time.time()}PPO",
        )
        self.model.set_logger(logger=self.logger)
        self.model.learn(
            total_timesteps=NUM_STEP_PER_EP * self.EVAL_EPS,
            callback=WandbCallback(
                verbose=1, model_save_path="./models/PPOEvaluator" + str(start_time)
            )
            if use_wandb
            else None,
        )
        end_time = time.time()
        seconds_elapsed = end_time - start_time
        print(f"Time elapsed: {str(datetime.timedelta(seconds=seconds_elapsed))}")


if __name__ == "__main__":
    import pickle

    np.random.seed(42)
    test = np.asarray(pickle.load(open("bw/LTE_test100kb.pickle", "rb")))  # Load file
    env = SinglepathEnvGym(bitrate_list=test)
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
