from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import argparse
import os
from stable_baselines3.common.env_util import make_vec_env
# from loguru import logger
from stable_baselines3.common.callbacks import CheckpointCallback
import imageio
from datetime import datetime
import numpy as np
import sys
import pathlib
import torch
import time
from typing import Union
from gymnasium import Env
from minigrid.wrappers import ImgObsWrapper



n_train_epoch = 2e6
env_name = "CustomDynamicObs-v0"

model_save_name = ("_".join(
        (
        str(env_name),
        str(n_train_epoch)
        )
        ))

# model = PPO.load("../model_save/" + model_save_name)
# model_save/CustomDynamicObs-v0_2000000.0.zip
# 
model = PPO.load("model_save/CustomDynamicObs-v0_2000000.0.zip")
# model = PPO.load("model_save/MiniGrid-Dynamic-Obstacles-6x6-v0_200000.0.zip")
render_mode = "rgb_array"
env = gym.make(env_name, render_mode=render_mode)
env = ImgObsWrapper(env)

# 环境中的位置列表
obs = env.reset()[0]
# print("obs:",obs)

obs = np.asarray(obs)
# print("type:",type(obs))
observations = []
# mode=render_mode
img = env.render()
images = []
# 采取的动作列表
actions = []
n_test_epochs = 2000

for i in range(n_test_epochs):
    images.append(img)
    action, _state = model.predict(obs, deterministic=True)
    # logger.info(f'action:{action}')
    print("_state:",_state)
    print("action:",action)
    actions.append(action)
    # obs, reward, terminated, truncated, info
    obs, reward, done, truncated, info = env.step(action)
    observations.append(obs)
    img = env.render()
    if done:
        break

        
now = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
gif_save_path = "logs/gif"
gif_fps = 1
imageio.mimsave(
        os.path.join(gif_save_path, f"{env_name}-{now}-{n_train_epoch}.gif"),
        [np.array(img) for i, img in enumerate(images) if i % 1 == 0],
        duration=gif_fps,
    )

# def prepare_testing(args):
#     logger.info(f"args: {args}")
#     args.gif_save_path = (
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/logs/gif"
#     )
#     args.model_save_path = (
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/models"
#     )
#     args.log_save_path = (
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/logs/tb_log"
#     )
#     args.checkpoints_save_path = (
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#         + "/models/checkpoints"
#     )
#     args.pth_save_freq = args.n_total_timesteps // 10
#     args.obstacle_size = args.lon_size * args.lat_size // 4

#     logger.warning(f"args: {args}")
#     os.makedirs(args.gif_save_path, exist_ok=True)
#     os.makedirs(args.model_save_path, exist_ok=True)
#     os.makedirs(args.log_save_path, exist_ok=True)
#     os.makedirs(args.checkpoints_save_path, exist_ok=True)


# def loading(args, env: Env):
#     # 加载模型
#     model_save_name = ("_".join(
#             (
#             str(env_name),
#             str(n_train_epoch)
#             )
#         ) + ".zip")
    
#     save_path = os.path.join(args.model_save_path, model_save_name)

#     if args.algo == "PPO":
#         model = PPO.load(save_path, env, verbose=1, tensorboard_log=args.log_save_path)
#     elif args.algo == "DQN":
#         model = DQN('MlpPolicy', env, verbose=1)
#         # save_path = '/home/gxy/ai_route/models/DQN_replay_buffer/seed_2_replay_buffer_steps_from_model.pkl'
#         model = DQN.load(save_path + '_seed_144', env, verbose=1, tensorboard_log=args.log_save_path)
#         # model = DQN('MlpPolicy', env, verbose=1)
#         model.save_replay_buffer('/home/gxy/ai_route/models/DQN_replay_buffer/' + 'seed_144_replay_buffer_steps')
#         # model.save_replay_buffer(os.path.join(args.model_save_path,'DQN_replay_buffer/seed_2_replay_buffer_steps'))
#         # model.load_replay_buffer(save_path)
#         # model.load_replay_buffer(os.path.join(args.model_save_path,'DQN_replay_buffer/dqn_replay_buffer'))
#     return model


# def testing(args):
#     now = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
#     # prepare for training
#     prepare_testing(args)
#     # record time
#     start_time = time.time()
#     # register env by env_id
#     env = gym.make(
#         args.env_id,
#         render_mode=args.render_mode,
#         lon_size=args.lon_size,
#         lat_size=args.lat_size,
#         obstacle_size=args.obstacle_size,
#         zest_size=args.zest_size,
#         seed = args.seed
#     )

#     # It will check your custom environment and output additional warnings if needed
#     check_env(env)

#     # init env params
#     params = {
#         "render_mode": args.render_mode,
#         "lon_size": args.lon_size,
#         "lat_size": args.lat_size,
#         "obstacle_size": args.obstacle_size,
#         "zest_size": args.zest_size,
#         "seed": args.seed,
#     }
#     # make DummyVectorEnv
#     train_env = make_vec_env(args.env_id, env_kwargs=params)
#     if train_env.num_envs != 1:
#         logger.warning("{} != 1 will cause some errors".format(train_env.num_envs))
#     # reset env
#     train_env.reset()
#     # load model
#     model = loading(args, train_env)

#     # 环境中的位置列表
#     obs = model.env.reset()
#     observations = []
#     # init render params
#     img = model.env.render(mode=args.render_mode)
#     images = []
#     # 采取的动作列表
#     actions = []
#     for i in range(args.n_test_epochs):
#         images.append(img)
#         action, _state = model.predict(obs, deterministic=True)
#         logger.info(f'action:{action}')
#         actions.append(action)
#         obs, reward, done, info = model.env.step(action)
#         observations.append(obs)
#         img = model.env.render(mode=args.render_mode)
#         if done:
#             break
#     process_time = str(round(time.time() - start_time, 4))

#     logger.info(
#         "Test Env: Cost Time: {}s, lon_size: {}, lat_size: {}, obstacle_size: {}, zest_size: {}, actions: {}, observations: {}".format(
#             process_time,
#             args.lon_size,
#             args.lat_size,
#             args.obstacle_size,
#             args.zest_size,
#             actions,
#             observations,
#         )
#     )

# testing(args=args)
