from PIL import Image
from collections import deque
from datetime import datetime
from pathlib import Path
import copy
import cv2
import imageio
import numpy as np
import random, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#import multiprocessing as mp
from torchvision import transforms as T

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.environment import *
from src.memory import *
from src.model import *

class Agent():
    def __init__(self, world, stage, action_type, envs, num_envs, state_dim, action_dim, save_dir, save_model_step,
                 save_figure_step, learn_step, total_step_or_episode, total_step, total_episode, model,
                 gamma, learning_rate, entropy_coef, V_coef, max_grad_norm,
                 clip_param, batch_size, num_epoch, is_normalize_advantage, V_loss_type, target_kl, gae_lambda, device):
        self.world = world
        self.stage = stage
        self.action_type = action_type

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.learn_step = learn_step
        self.total_step_or_episode = total_step_or_episode
        self.total_step = total_step
        self.total_episode = total_episode

        self.current_step = 0
        self.current_episode = 0

        self.save_model_step = save_model_step
        self.save_figure_step = save_figure_step

        self.device = device
        self.save_dir = save_dir

        self.num_envs = num_envs
        self.envs = envs
        self.model = model.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.V_coef = V_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.num_epoch = num_epoch

        self.memory = Memory(self.num_envs)
        self.is_completed = False

        self.env = None
        self.max_test_score = -1e9
        self.is_normalize_advantage = is_normalize_advantage
        self.V_loss_type = V_loss_type
        self.target_kl = target_kl
        self.gae_lambda = gae_lambda

        # I just log 1000 lastest update and print it to log.
        self.V_loss = np.zeros((1000,)).reshape(-1)
        self.P_loss = np.zeros((1000,)).reshape(-1)
        self.E_loss = np.zeros((1000,)).reshape(-1)
        self.approx_kl_divs = np.zeros((1000,)).reshape(-1)
        self.total_loss = np.zeros((1000,)).reshape(-1)
        self.loss_index = 0
        self.len_loss = 0

    def save_figure(self, is_training=False):
        # test current model and save model/figure if model yield best total rewards.
        # create env for testing, reset test env
        if self.env is None:
            self.env = create_env(self.world, self.stage, self.action_type, True)
        state = self.env.reset()
        done = False

        images = []
        total_reward = 0
        total_step = 0
        num_repeat_action = 0
        old_action = -1

        episode_time = datetime.now()

        # play 1 episode, just get loop action with max probability from model until the episode end.
        while not done:
            with torch.no_grad():
                logit, V = self.model(torch.tensor(state, dtype = torch.float, device = self.device).unsqueeze(0))
            action = logit.argmax(-1).item()
            next_state, reward, done, trunc, info = self.env.step(action)
            state = next_state
            img = Image.fromarray(self.env.current_state)
            images.append(img)
            total_reward += reward
            total_step += 1

            if action == old_action:
                num_repeat_action += 1
            else:
                num_repeat_action = 0
            old_action = action
            if num_repeat_action == 200:
                break

        #logging, if model yield better result, save figure (test_episode.mp4) and model (best_model.pth)
        if is_training:
            f_out = open(f"logging_test.txt", "a")
            f_out.write(f'episode_reward: {total_reward:.4f} episode_step: {total_step} current_step: {self.current_step} loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} loss: {(self.total_loss.sum()/self.len_loss):.4f} approx_kl_div: {(self.approx_kl_divs.sum()/self.len_loss):.4f} episode_time: {datetime.now() - episode_time}\n')
            f_out.close()

        if total_reward > self.max_test_score or info['flag_get']:
            imageio.mimsave('test_episode.mp4', images)
            self.max_test_score = total_reward
            if is_training:
                torch.save(self.model.state_dict(), f"best_model.pth")

        # if model can complete this game, stop training by set self.is_completed to True
        if info['flag_get']:
            self.is_completed = True

    def save_model(self):
        torch.save(self.model.state_dict(), f"model_{self.current_step}.pth")

    def load_model(self, model_path = None):
        if model_path is None:
            model_path = f"model_{self.current_step}.pth"
        self.model.load_state_dict(torch.load(model_path))

    def update_loss_statis(self, loss_p, loss_v, loss_e, loss, approx_kl_div):
        # update loss for logging, just save 1000 latest updates.
        self.V_loss[self.loss_index] = loss_v
        self.P_loss[self.loss_index] = loss_p
        self.E_loss[self.loss_index] = loss_e
        self.total_loss[self.loss_index] = loss
        self.approx_kl_divs[self.loss_index] = approx_kl_div
        self.loss_index = (self.loss_index + 1)%1000
        self.len_loss = min(self.len_loss+1, 1000)

    def select_action(self, states):
        # select action when training, we need use Categorical distribution to make action base on probability from model
        states = torch.tensor(np.array(states), device = self.device)

        with torch.no_grad():
            logits, V = self.model(states)
            policy = F.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(policy)
            actions = distribution.sample().cpu().numpy().tolist()
        return actions, logits, V

    def learn(self):
        # get all data
        states, actions, next_states, rewards, dones, old_logits, old_values = self.memory.get_data()

        # calculate target (td lambda target) and gae advantages
        targets = []
        with torch.no_grad():
            _, next_value = self.model(torch.tensor(np.array(next_states[-1]), device = self.device))
        target = next_value
        advantage = 0

        for state, next_state, reward, done, V in zip(states[::-1], next_states[::-1], rewards[::-1], dones[::-1], old_values[::-1]):
            done = torch.tensor(done, device = self.device, dtype = torch.float).reshape(-1, 1)
            reward = torch.tensor(reward, device = self.device).reshape(-1, 1)

            target = next_value * self.gamma * (1-done) + reward
            advantage = target + self.gamma * advantage * (1-done) * self.gae_lambda
            targets.append(advantage)
            advantage = advantage - V.detach()
            next_value = V.detach()
        targets = targets[::-1]

        # convert all data to tensor
        action_index = torch.flatten(torch.tensor(actions, device = self.device, dtype = torch.int64))
        states = torch.tensor(np.array(states), device = self.device)
        states = states.reshape((-1,  states.shape[2], states.shape[3], states.shape[4]))
        old_values = torch.cat(old_values, 0)
        targets = torch.cat(targets, 0).view(-1, 1)
        old_logits = torch.cat(old_logits, 0)
        old_probs = torch.softmax(old_logits, -1)
        index = torch.arange(0, len(old_probs), device = self.device)
        old_log_probs = (old_probs[index, action_index] + 1e-9).log()
        advantages = (targets - old_values).reshape(-1)
        early_stopping = False

        #train num_epoch time
        for epoch in range(self.num_epoch):
            #shuffle data for each epoch
            shuffle_ids = torch.randperm(len(targets), dtype = torch.int64)
            for i in range(len(old_values)//self.batch_size):
                #train with batch_size data
                self.optimizer.zero_grad()
                start_id = i * self.batch_size
                end_id = min(len(shuffle_ids), (i+1) * self.batch_size)
                batch_ids = shuffle_ids[start_id:end_id]

                #predict logits and values from model
                logits, values = self.model(states[batch_ids])

                #calculate entropy and value loss (using mse or huber based on config)
                probs =  torch.softmax(logits, -1)
                entropy = (- (probs * (probs + 1e-9).log()).sum(-1)).mean()
                if self.V_loss_type == 'huber':
                    loss_V = F.smooth_l1_loss(values, targets[batch_ids])
                else:
                    loss_V = F.mse_loss(values, targets[batch_ids])

                # calculate log_probs
                index = torch.arange(0, len(probs), device = self.device)
                batch_action_index = action_index[batch_ids]

                log_probs = (probs[index, batch_action_index] + 1e-9).log()

                #approx_kl_div copy from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO
                #if approx_kl_div larger than 1.5 * target_kl (if target_kl in config is not None), stop training because policy change so much
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[batch_ids]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    early_stopping = True

                #calculate policy loss
                ratio = torch.exp(log_probs - old_log_probs[batch_ids])
                batch_advantages = advantages[batch_ids].detach()
                if self.is_normalize_advantage:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-9)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                loss_P = -torch.min(surr1, surr2).mean()

                # update model
                loss = - entropy * self.entropy_coef + loss_V * self.V_coef + loss_P

                self.update_loss_statis(loss_P.item(), loss_V.item(), entropy.item(), loss.item(), approx_kl_div.item())

                if early_stopping == False:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else:
                    break
            if early_stopping:
                break

    def train(self):
        episode_reward = [0] * self.num_envs
        episode_step = [0] * self.num_envs
        max_episode_reward = 0
        max_episode_step = 0
        episode_time = [datetime.now() for _ in range(self.num_envs)]
        total_time = datetime.now()

        last_episode_rewards = []

        #reset envs
        states = self.envs.reset()

        while True:
            # finish training if agent reach total_step or total_episode base on what type of total_step_or_episode is step or episode
            self.current_step += 1

            if self.total_step_or_episode == 'step':
                if self.current_step >= self.total_step:
                    break
            else:
                if self.current_episode >= self.total_episode:
                    break

            actions, logits, values = self.select_action(states)

            next_states, rewards, dones, truncs, infos = self.envs.step(actions)

            # save to memory
            self.memory.save(states, actions, rewards, next_states, dones, logits, values)

            episode_reward = [x + reward for x, reward in zip(episode_reward, rewards)]
            episode_step = [x+1 for x in episode_step]

             # logging after each step, if 1 episode is ending, I will log this to logging.txt
            for i, done in enumerate(dones):
                if done:
                    self.current_episode += 1
                    max_episode_reward = max(max_episode_reward, episode_reward[i])
                    max_episode_step = max(max_episode_step, episode_step[i])
                    last_episode_rewards.append(episode_reward[i])
                    f_out = open(f"logging.txt", "a")
                    f_out.write(f'episode: {self.current_episode} agent: {i} rewards: {episode_reward[i]:.4f} steps: {episode_step[i]} complete: {infos[i]["flag_get"]==True} mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean():.4f} max_rewards: {max_episode_reward:.4f} max_steps: {max_episode_step} current_step: {self.current_step} loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} loss: {(self.total_loss.sum()/self.len_loss):.4f} approx_kl_div: {(self.approx_kl_divs.sum()/self.len_loss):.4f} episode_time: {datetime.now() - episode_time[i]} total_time: {datetime.now() - total_time}\n')
                    f_out.close()
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_time[i] = datetime.now()

            # training agent every learn_step
            if self.current_step % self.learn_step == 0:
                self.learn()
                self.memory.reset()

            # eval agent every save_figure_step
            if self.current_step % self.save_figure_step == 0:
                self.save_figure(is_training=True)
                if self.is_completed:
                    return

            if self.current_step % self.save_model_step == 0:
                self.save_model()

            states = list(next_states)

        f_out = open(f"logging.txt", "a")
        f_out.write(f' mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean()} max_rewards: {max_episode_reward} max_steps: {max_episode_step} current_step: {self.current_step} total_time: {datetime.now() - total_time}\n')
        f_out.close()