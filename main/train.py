import gymnasium as gym

import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,CheckpointCallback,EvalCallback,StopTrainingOnRewardThreshold
import time
import os


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

start_time = time.time()

n_train_epoch = 1e5
env_name = "CustomDynamicObs-v0"
env = gym.make(env_name, render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)


# pth_save_freq = 1e3
# checkpoints_save_path = "logs/checkpoints"
# log_save_path = "logs/tb_log"
# model_save_name = ("_".join(
#         (
#         str(env_name),
#         str(n_train_epoch)
#         )
#         ) + ".zip")

# checkpoint_callback = CheckpointCallback(
#     save_freq=pth_save_freq,
#     save_path=checkpoints_save_path,
#     name_prefix=env_name + '_' + model_save_name,
# )


model.learn(n_train_epoch)

model.save("/Users/gaoxuanyu/Documents/Minigrid-master/model_save/" + model_save_name)

end_time = time.time()

print(f"------running time: {end_time - start_time} s ----------")

print("---------- finish ----------")