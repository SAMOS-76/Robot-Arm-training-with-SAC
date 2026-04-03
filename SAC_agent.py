import random 
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import LinearLR
from collections import deque

# Actor takes in current state and produces mean and standard deviation of a normal distribution
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Separating outputs for mean and log_sd # Why?
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_sd_layer = nn.Linear(hidden, act_dim)

    def forward(self, x):
        x = self.network(x)
        mean = self.mean_layer(x)
        log_sd = self.log_sd_layer(x)
        # clamp so values aren't too large or small
        log_sd = torch.clamp(log_sd, min=-20, max=2)
        return mean, log_sd

# We pass into our Critic neworks
class CriticNetworks(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        self.network1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.network2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        q1 = self.network1(x)
        q2 = self.network2(x)
        return q1, q2
    
class CNNEncoder(nn.Module):
    def __init__(self, image_shape, hidden=10, latent_dim=50):
        super().__init__()

        height, width, depth = image_shape

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        with torch.no_grad():
            empty_conv = torch.zeros(1, depth, height, width)
            conv_out = self.layer_2(self.layer_1(empty_conv))
            flat_dim = conv_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flat_dim, out_features=latent_dim)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        return self.classifier(x)
    
class SACAgent():
    def __init__(self, env, device, timesteps = 1000000):
        self.env = env
        self.device = device
        self.total_timesteps = timesteps

        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.0003
        self.gamma = 0.995
        self.batch_size = 256
        self.replay_size = 30000
        self.learning_steps = 5000

        self.latent_dim = 50

        joint_dim = env.single_observation_space["joints"].shape[0]
        act_dim = env.single_action_space.shape[0]
        self.fused_obs_dim = self.latent_dim + joint_dim

        image_shape = env.single_observation_space["image"].shape
        self.encoder = CNNEncoder(image_shape, latent_dim=self.latent_dim).to(self.device)
        self.Critic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)
        self.Actor = ActorNetwork(self.fused_obs_dim, act_dim).to(self.device)
        self.TargetCritic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)
        self.TargetCritic.load_state_dict(self.Critic.state_dict())
        for parameter in self.TargetCritic.parameters():
            parameter.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optim = torch.optim.Adam(list(self.Critic.parameters()) + list(self.encoder.parameters()), lr=self.critic_learning_rate)

        # We optimize the log of alpha to ensure alpha always remains positive
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0003)
        self.target_entropy = -act_dim

    def fuse_observations(self, obs, detach_encoder=False): # detach encoder for actor
        images = torch.as_tensor(obs["image"], device=self.device)
        joints = torch.as_tensor(obs["joints"], dtype=torch.float32, device=self.device)

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if joints.ndim == 1:
            joints = joints.unsqueeze(0)

        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2).contiguous()

        latent_image = self.encoder(images)
        if detach_encoder:
            latent_image = latent_image.detach()

        fused_obs = torch.cat([latent_image, joints], dim=1)
        return fused_obs


    def update_critic(self, next_obs, obs, action, reward, done):
        with torch.no_grad():
            # Get "future" action
            f_mean, f_log_sd = self.Actor(next_obs)
            sd = f_log_sd.exp()
            normal = Normal(f_mean, sd)
            x = normal.rsample()
            future_action = torch.tanh(x)
            # Get log probs
            log_prob = normal.log_prob(x)
            log_prob -= torch.log(1 - future_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            # Pass "next" state and action into target networks to get our "true" value of the state
            q_value1, q_value2 = self.TargetCritic(next_obs, future_action)
            q_value = torch.minimum(q_value1, q_value2)
            # Bellman target
            alpha = self.log_alpha.exp().detach()
            b_target = reward + self.gamma*(1-done)*(q_value - alpha*log_prob)

        q_value1, q_value2 = self.Critic(obs, action)
        
        critic_loss = torch.nn.functional.mse_loss(q_value1, b_target).mean() + torch.nn.functional.mse_loss(q_value2, b_target).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def update_actor(self, obs):
        mean, log_sd = self.Actor(obs)
        sd = log_sd.exp()
        normal = Normal(mean, sd)
        x = normal.rsample()
        action = torch.tanh(x)
        # Get log probs
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        for param in self.Critic.parameters():
            param.requires_grad = False

        q_value1, q_value2 = self.Critic(obs, action)
        q_value = torch.minimum(q_value1, q_value2)

        for param in self.Critic.parameters():
            param.requires_grad = True

        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha*log_prob - q_value).mean()


        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

    def train(self, model_path, save_timesteps = 20000):
        n_envs = self.env.num_envs
        episode_rewards = np.zeros(n_envs)
        episode_returns = []
        episode_count = 0
        tau = 0.005

        success_history = deque(maxlen=100)
        final_distance_history = deque(maxlen=100)
        buffer = deque(maxlen=self.replay_size)

        raw_obs, _ = self.env.reset()
        with torch.no_grad():
            fused_obs = self.fuse_observations(raw_obs, detach_encoder=True)

        for global_step in range(self.total_timesteps):
            with torch.no_grad():
                mean, log_sd = self.Actor(fused_obs)
                sd = log_sd.exp()
                normal = Normal(mean, sd)
                x = normal.rsample()
                action = torch.tanh(x)
                actions_np = action.cpu().numpy()

            raw_next_obs, reward, terminated, truncated, infos = self.env.step(actions_np)
            with torch.no_grad():
                fused_next_obs = self.fuse_observations(raw_next_obs)

            dones = np.logical_or(terminated, truncated)
            episode_rewards += reward

            step_success = np.full(n_envs, np.nan, dtype=np.float32)
            step_distance = np.full(n_envs, np.nan, dtype=np.float32)

            if "success" in infos:
                step_success = np.asarray(infos["success"], dtype=np.float32)

            if "distance" in infos:
                step_distance = np.asarray(infos["distance"], dtype=np.float32)

            if "final_info" in infos and "_final_info" in infos:
                final_infos = infos["final_info"]
                final_mask = infos["_final_info"]
                for i in range(n_envs):
                    if final_mask[i] and final_infos[i] is not None:
                        fi = final_infos[i]
                        if "success" in fi:
                            step_success[i] = 1.0 if bool(fi["success"]) else 0.0
                        if "distance" in fi:
                            step_distance[i] = float(fi["distance"])

            for i in range(n_envs):
                obs = {
                    "image": raw_obs["image"][i].copy(),
                    "joints": raw_obs["joints"][i].astype(np.float32, copy=True),
                }
                next_obs = {
                    "image": raw_next_obs["image"][i].copy(),
                    "joints": raw_next_obs["joints"][i].astype(np.float32, copy=True),
                }

                if "final_observation" in infos and infos.get("_final_observation", [False] * n_envs)[i]:
                    final_obs = infos["final_observation"][i]
                    next_obs = {
                        "image": final_obs["image"].copy(),
                        "joints": final_obs["joints"].astype(np.float32, copy=True),
                    }

                buffer.append((obs, actions_np[i], reward[i], next_obs, dones[i]))

            # for i in range(n_envs):
            #     if dones[i]:
            #         ret = float(episode_rewards[i])
            #         episode_returns.append(ret)
            #         episode_count += 1
            #         print(f"global_step={global_step + i}, episode={episode_count}, episode_return={ret}")
            #         episode_rewards[i] = 0

            for i in range(n_envs):
                if dones[i]:
                    ret = float(episode_rewards[i])
                    episode_returns.append(ret)
                    episode_count += 1

                    success_i = step_success[i]
                    if np.isnan(success_i):
                        success_i = 1.0 if terminated[i] else 0.0
                    success_history.append(float(success_i))

                    dist_i = step_distance[i]
                    if not np.isnan(dist_i):
                        final_distance_history.append(float(dist_i))

                    ret10 = float(np.mean(episode_returns[-10:])) if len(episode_returns) >= 1 else float("nan")
                    succ100 = float(np.mean(success_history)) if len(success_history) > 0 else float("nan")
                    dist100 = float(np.mean(final_distance_history)) if len(final_distance_history) > 0 else float("nan")

                    print(
                        f"step={global_step + i}, episode={episode_count}, "
                        f"episode_return={ret:.3f}, return10={ret10:.3f}, "
                        f"success100={succ100:.3f}, final_dist100={dist100:.4f}"
                    )

                    episode_rewards[i] = 0.0

            # Training
            if len(buffer) > self.learning_steps:
                batch = random.sample(buffer, k=self.batch_size)
                obs_raw_, actions_, rewards_, next_obs_raw_, dones_ = zip(*batch)

                obs_batch = {
                    "image": np.stack([o["image"] for o in obs_raw_], axis=0),
                    "joints": np.stack([o["joints"] for o in obs_raw_], axis=0),
                }
                next_obs_batch = {
                    "image": np.stack([o["image"] for o in next_obs_raw_], axis=0),
                    "joints": np.stack([o["joints"] for o in next_obs_raw_], axis=0),
                }

                tensor_obs = self.fuse_observations(obs_batch, detach_encoder=False)
                tensor_next_obs = self.fuse_observations(next_obs_batch, detach_encoder=False)
                tensor_actions = torch.as_tensor(np.array(actions_), dtype=torch.float32, device=self.device)
                tensor_reward = torch.as_tensor(np.array(rewards_), dtype=torch.float32, device=self.device).view(-1, 1)
                tensor_dones = torch.as_tensor(np.array(dones_), dtype=torch.float32, device=self.device).view(-1, 1)

                # Training Critic
                self.update_critic(tensor_next_obs, tensor_obs, tensor_actions, tensor_reward, tensor_dones)

                # Train Actor: need to compare to newly updated critic
                self.update_actor(tensor_obs.detach())

                for target_param, local_param in zip(self.TargetCritic.parameters(), self.Critic.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
            raw_obs = raw_next_obs
            fused_obs = fused_next_obs.detach()

            if global_step % save_timesteps == 0:
                torch.save(self.Actor.state_dict(), f"{model_path}/{global_step}")

            if len(episode_returns) > 10 and np.mean(episode_returns[-10:]) >= 0:
                print("max reward achieved")
                torch.save(self.Actor.state_dict(), f"{model_path}/{global_step}")
                break