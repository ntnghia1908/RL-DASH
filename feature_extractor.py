import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Categorical
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PensieveFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(PensieveFeatureExtractor, self).__init__(observation_space, features_dim)
        self.network_speed = nn.Sequential(
            nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten()
        )
        self.next_chunk_size = nn.Sequential(
            nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten()
        )
        self.buffer_size = nn.Sequential(nn.Linear(1, 128), nn.ReLU())
        self.percentage_remain_video_chunks = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU()
        )
        self.last_play_quality = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(), nn.Flatten()
        )
        self.delay_net = nn.Sequential(nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten())
        self.last_layer = nn.Sequential(
            nn.Linear(1664, self.features_dim * 2),
            nn.Tanh(),
            nn.Linear(self.features_dim * 2, self.features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        network_speed = self.network_speed(observations["network_speed"].unsqueeze(-2))
        next_chunk_size = self.next_chunk_size(
            observations["next_chunk_size"].unsqueeze(-2)
        )
        buffer_size = self.buffer_size(observations["buffer_size"])
        percentage_remain_video_chunks = self.percentage_remain_video_chunks(
            observations["percentage_remain_video_chunks"]
        )
        last_down_quality = self.last_play_quality(observations["last_down_quality"])
        delay = self.delay_net(observations["delay"].unsqueeze(-2))
        cat = torch.cat(
            (
                network_speed,
                next_chunk_size,
                buffer_size,
                percentage_remain_video_chunks,
                last_down_quality,
                delay,
            ),
            dim=1,
        )
        out = self.last_layer(cat)
        return out
