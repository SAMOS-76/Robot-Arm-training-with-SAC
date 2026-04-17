import os 
import datetime
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import LinearLR
from collections import deque, namedtuple

StepInfo = namedtuple('StepInfo', ['obs_raw', 'action', 'reward', 'next_obs_raw', 'done'])

# Actor takes in current state and produces mean and standard deviation of a normal distribution
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
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
        # clamp so values aren't too large or small giving infinity or NaN
        log_sd = torch.clamp(log_sd, min=-20, max=2)
        return mean, log_sd

"""
We pass into our Critic neworks
Two critic Networks in one module since we want to easily update them with the same loss
"""
class CriticNetworks(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        self.network1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.network2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        q1 = self.network1(x)
        q2 = self.network2(x)
        return q1, q2
    
# Image encoder for specific robot implementation with camera
# Need to compress image into latent space embedding to be concatenated with robot observations
# NOTE: removed encoder to speed up training
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
    
# SAC class
class SACAgent():
    def __init__(self, env, device, timesteps=1000000, use_images=False, replay_size=1_000_000, batch_size=512, updates_per_step=2):
        self.env = env
        self.device = device
        self.total_timesteps = timesteps
        self.use_images = bool(use_images)

        # Hyperparameters
        self.critic_learning_rate = 0.0001
        self.actor_learning_rate = 0.0001
        self.gamma = 0.995
        self.batch_size = int(batch_size)
        self.replay_size = int(replay_size)
        self.updates_per_step = max(1, int(updates_per_step))
        self.learning_steps = 50_000
        self.random_explore_steps = 50_000
        self.latent_dim = 50

        # Transition-capped episodic replay 
        self.replay_buffer = GlobalEpisodicReplayBuffer(max_transitions=self.replay_size)

        joint_dim = env.single_observation_space["joints"].shape[0]
        act_dim = env.single_action_space.shape[0]

        # To switch between learning from video or not
        if self.use_images:
            image_shape = env.single_observation_space["image"].shape
            self.encoder = CNNEncoder(image_shape, latent_dim=self.latent_dim).to(self.device)
            self.fused_obs_dim = self.latent_dim + joint_dim
        else:
            self.encoder = None
            self.fused_obs_dim = joint_dim

        # Network Initialisation
        self.Critic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)
        self.Actor = ActorNetwork(self.fused_obs_dim, act_dim).to(self.device)
        self.TargetCritic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)

        self.TargetCritic.load_state_dict(self.Critic.state_dict())
        for parameter in self.TargetCritic.parameters():
            parameter.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=self.actor_learning_rate)

        critic_params = list(self.Critic.parameters())
        if self.encoder is not None:
            critic_params.extend(list(self.encoder.parameters()))
        self.critic_optim = torch.optim.Adam(critic_params, lr=self.critic_learning_rate)

        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0003)
        self.target_entropy = -act_dim

        self.log_every_episodes = 10

        self.grab_tolerance = 0.025
        self.stage_tolerance = 0.03
        self.success_tolerance = 0.02
        self.task_stage = 4
        self._sync_tolerances_from_env()

    def _sync_tolerances_from_env(self):
        def _read_env_scalar(attr_name):
            if hasattr(self.env, attr_name):
                try:
                    return float(getattr(self.env, attr_name))
                except (TypeError, ValueError):
                    pass

            if hasattr(self.env, "get_attr"):
                try:
                    values = self.env.get_attr(attr_name)
                    if values and len(values) > 0:
                        return float(values[0])
                except Exception:
                    return None

            return None

        env_grab_tol = _read_env_scalar("grab_tolerance")
        env_stage_tol = _read_env_scalar("stage_tolerance")
        env_success_tol = _read_env_scalar("success_tolerance")
        env_task_stage = _read_env_scalar("task_stage")

        if env_grab_tol is not None:
            self.grab_tolerance = env_grab_tol
        if env_stage_tol is not None:
            self.stage_tolerance = env_stage_tol
        if env_success_tol is not None:
            self.success_tolerance = env_success_tol
        if env_task_stage is not None:
            self.task_stage = int(env_task_stage)

    # Fuse joint and image observations if using image input
    def fuse_observations(self, obs, detach_encoder=False):
        joints = torch.as_tensor(obs["joints"], dtype=torch.float32, device=self.device)
        if joints.ndim == 1:
            joints = joints.unsqueeze(0)

        if not self.use_images:
            return joints

        images = torch.as_tensor(obs["image"], device=self.device)
        if images.ndim == 3:
            images = images.unsqueeze(0)

        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2).contiguous()

        latent_image = self.encoder(images)
        if detach_encoder:
            latent_image = latent_image.detach()

        fused_obs = torch.cat([latent_image, joints], dim=1)
        return fused_obs

    # Critic learning
    def update_critic(self, next_obs, obs, action, reward, done):
        with torch.no_grad():
            # Get "future" action by passing in next_obs from replay buffer
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

        return critic_loss.item()

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

        # Don't want to update critic while training Actor
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

        # Update alpha during Actor loop 
        # Below was very unstable changed to alpha trick
        # alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return actor_loss.item(), alpha_loss.item()

    def _compute_reward_and_success(self, gripper_pos, red_block_pos, blue_block_pos, target_bottom_pos, target_top_pos, action=None):
        dense_coef = 0.30
        dense_scale = 8.0

        # REACH
        if self.task_stage == 1:
            goal_dist = np.linalg.norm(gripper_pos - target_bottom_pos)
            is_success = bool(goal_dist < self.success_tolerance)

        # PUSH / PICK & PLACE
        elif self.task_stage in (2, 3):
            dist_gripper_to_red = np.linalg.norm(gripper_pos - red_block_pos)
            dist_red_to_target = np.linalg.norm(red_block_pos - target_bottom_pos)
            
            # Agent is rewarded for moving gripper to block, AND block to target.
            goal_dist = dist_gripper_to_red + dist_red_to_target
            is_success = bool(dist_red_to_target < self.success_tolerance)

        # STACK
        else:
            # Assuming red goes to bottom target, blue goes to top target.
            # You might also want a gripper_to_blue term here depending on task complexity.
            dist_red = np.linalg.norm(red_block_pos - target_bottom_pos)
            dist_blue = np.linalg.norm(blue_block_pos - target_top_pos)
            
            # Sum distances so gradients flow for both blocks simultaneously.
            goal_dist = (dist_red + dist_blue) / 2.0 
            is_success = bool(
                (dist_red < self.success_tolerance) and (dist_blue < self.success_tolerance)
            )

        # Sparse terminal reward + small HER-friendly dense shaping.
        if is_success:
            reward = 0.0
        else:
            # Exponential shaping bounds the penalty nicely.
            reward = -1.0 + dense_coef * np.exp(-dense_scale * goal_dist)

        return float(reward), is_success

        
    def sample(self):
        if self.replay_buffer.get_total_episodes() == 0:
            return None
        if self.replay_buffer.get_total_timesteps() < self.batch_size:
            return None
        
        episodes = list(self.replay_buffer.buffer) # To make mutable

        # Weighted sampling to prioritise sampling from longer eps
        # lengths = np.asarray(self.replay_buffer.episode_lengths, dtype=np.float64)
        # probs = lengths / lengths.sum()
        # episode_indices = np.random.choice(len(episodes), size=self.batch_size, replace=True, p=probs) 

        # Uniform sampling so it doesn't emphasise possible failed trajectories at end of episodes
        episode_indices = np.random.choice(len(episodes), size=self.batch_size, replace=True)
        HER_masking = np.random.rand(self.batch_size) < 0.8

        obs_batch = []
        actions_batch = []
        rewards_batch = []
        next_obs_batch = []
        dones_batch = []

        for index, episode_id in enumerate(episode_indices):
            episode = episodes[episode_id]
            length = len(episode)
            if HER_masking[index] and length > 1:
                step = np.random.randint(0, length - 1)  # Error checking to guarantee future step
            else:
                step = np.random.randint(0, length)
            timestep = episode[step]
            obs_joints = timestep.obs_raw.copy()
            next_obs_joints = timestep.next_obs_raw.copy()
            action = np.asarray(timestep.action, dtype=np.float32).copy()
            reward = float(timestep.reward)
            done = float(timestep.done)

            if HER_masking[index] and length > 1:
                future_timestep = np.random.randint(step + 1, length)
                future_obs = episode[future_timestep].next_obs_raw
                new_goal = next_obs_joints[-6:].copy()

                if self.task_stage == 1:
                    new_goal[:3] = future_obs[-15:-12].copy()
                    new_goal[3:] = 0.0
                elif self.task_stage in (2, 3):
                    new_goal[:3] = future_obs[-12:-9].copy()
                    new_goal[3:] = 0.0
                else:
                    new_goal[:3] = future_obs[-12:-9].copy()
                    new_goal[3:] = future_obs[-9:-6].copy()

                # Change old goals with new goals
                obs_joints[-6:] = new_goal
                next_obs_joints[-6:] = new_goal
                # Cause I've changed the goals I also need to change the distance vectors in the obs
                if self.task_stage == 1:
                    obs_joints[15:18] = 0.0
                    next_obs_joints[15:18] = 0.0
                else:
                    obs_joints[15:18] = new_goal[:3] - obs_joints[-12:-9]
                    next_obs_joints[15:18] = new_goal[:3] - next_obs_joints[-12:-9]

                if self.task_stage == 4:
                    obs_joints[21:24] = new_goal[3:] - obs_joints[-9:-6]
                    next_obs_joints[21:24] = new_goal[3:] - next_obs_joints[-9:-6]
                else:
                    obs_joints[21:24] = 0.0
                    next_obs_joints[21:24] = 0.0

                reward, success = self._compute_reward_and_success(
                    gripper_pos=next_obs_joints[-15:-12],
                    red_block_pos=next_obs_joints[-12:-9],
                    blue_block_pos=next_obs_joints[-9:-6],
                    target_bottom_pos=new_goal[:3],
                    target_top_pos=new_goal[3:],
                    action=action,
                )

                reward = float(reward)
                if success:
                    done = 1.0

            obs_batch.append(obs_joints.astype(np.float32, copy=False))
            actions_batch.append(action)
            rewards_batch.append(np.float32(reward))
            next_obs_batch.append(next_obs_joints.astype(np.float32, copy=False))
            dones_batch.append(np.float32(done))

        return (np.stack(obs_batch, axis=0), np.stack(actions_batch, axis=0), np.asarray(rewards_batch, dtype=np.float32), np.stack(next_obs_batch, axis=0), np.asarray(dones_batch, dtype=np.float32))

    # Modified existing load checkpoint code
    # NOTE: Should I load a new critic and alpha each stage? Difficult for critic to unlearn if the environment states are too different
    # def load_checkpoint(self, model_path, timestep, load_critic=True):
    #     actor_path = os.path.join(model_path, "Actor", str(timestep))
    #     critic_path = os.path.join(model_path, "Critic", str(timestep))
    #     encoder_path = os.path.join(model_path, "Encoder", str(timestep))
    #     alpha_path = os.path.join(model_path, "Alpha", str(timestep))

    #     if not os.path.isfile(actor_path):
    #         raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")

    #     actor_state = torch.load(actor_path, map_location=self.device)
    #     self.Actor.load_state_dict(actor_state)

    #     if load_critic:
    #         if not os.path.isfile(critic_path):
    #             raise FileNotFoundError(f"Critic checkpoint not found: {critic_path}")
    #         critic_state = torch.load(critic_path, map_location=self.device)
    #         self.Critic.load_state_dict(critic_state)
    #         self.TargetCritic.load_state_dict(self.Critic.state_dict())

    #     if self.encoder is not None:
    #         if os.path.isfile(encoder_path):
    #             encoder_state = torch.load(encoder_path, map_location=self.device)
    #             self.encoder.load_state_dict(encoder_state)
    #         else:
    #             print(f"[WARN] Encoder checkpoint not found: {encoder_path}. Using current encoder weights.")

    #     if load_critic and os.path.isfile(alpha_path):
    #         alpha_state = torch.load(alpha_path, map_location=self.device)
    #         if "log_alpha" in alpha_state:
    #             self.log_alpha.data.copy_(alpha_state["log_alpha"].to(self.device))
    #         if "alpha_optim" in alpha_state:
    #             self.alpha_optim.load_state_dict(alpha_state["alpha_optim"])

    #     print(f"Loaded checkpoint at timestep {timestep}")

    def load_checkpoint(self, model_path, timestep, load_critic=True):
        actor_dir = os.path.join(model_path, "Actor")
        critic_dir = os.path.join(model_path, "Critic")
        encoder_dir = os.path.join(model_path, "Encoder")
        alpha_dir = os.path.join(model_path, "Alpha")

        def _resolve_checkpoint_file(folder, step):
            # Prefer solved file, then plain, then interrupted.
            preferred = [f"{step}_solved", str(step), f"{step}_interrupted"]
            for name in preferred:
                path = os.path.join(folder, name)
                if os.path.isfile(path):
                    return path

            # Fallback: any file starting with the numeric step.
            prefix = f"{step}"
            candidates = []
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    if not name.startswith(prefix):
                        continue
                    path = os.path.join(folder, name)
                    if os.path.isfile(path):
                        # Prefer solved > plain > interrupted > other.
                        if name.endswith("_solved"):
                            rank = 3
                        elif name == str(step):
                            rank = 2
                        elif name.endswith("_interrupted"):
                            rank = 1
                        else:
                            rank = 0
                        candidates.append((rank, name, path))

            if not candidates:
                return None

            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return candidates[0][2]

        actor_path = _resolve_checkpoint_file(actor_dir, timestep)
        if actor_path is None:
            raise FileNotFoundError(f"Actor checkpoint not found for step {timestep} in: {actor_dir}")

        checkpoint_name = os.path.basename(actor_path)
        critic_path = os.path.join(critic_dir, checkpoint_name)
        encoder_path = os.path.join(encoder_dir, checkpoint_name)
        alpha_path = os.path.join(alpha_dir, checkpoint_name)

        actor_state = torch.load(actor_path, map_location=self.device)
        self.Actor.load_state_dict(actor_state)

        if load_critic:
            if not os.path.isfile(critic_path):
                raise FileNotFoundError(f"Critic checkpoint not found: {critic_path}")
            critic_state = torch.load(critic_path, map_location=self.device)
            self.Critic.load_state_dict(critic_state)
            self.TargetCritic.load_state_dict(self.Critic.state_dict())

        if self.encoder is not None:
            if os.path.isfile(encoder_path):
                encoder_state = torch.load(encoder_path, map_location=self.device)
                self.encoder.load_state_dict(encoder_state)
            else:
                print(f"[WARN] Encoder checkpoint not found: {encoder_path}. Using current encoder weights.")

        if load_critic and os.path.isfile(alpha_path):
            alpha_state = torch.load(alpha_path, map_location=self.device)
            if "log_alpha" in alpha_state:
                self.log_alpha.data.copy_(alpha_state["log_alpha"].to(self.device))
            if "alpha_optim" in alpha_state:
                self.alpha_optim.load_state_dict(alpha_state["alpha_optim"])

        print(f"Loaded checkpoint: {checkpoint_name}")

    def train(self, model_path, save_timesteps=50000, start_timestep=0, target_success_rate=0.9):
        task_solved = False
        actor_dir = os.path.join(model_path, "Actor")
        critic_dir = os.path.join(model_path, "Critic")
        encoder_dir = os.path.join(model_path, "Encoder")
        alpha_dir = os.path.join(model_path, "Alpha")
        os.makedirs(alpha_dir, exist_ok=True)
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(critic_dir, exist_ok=True)
        if self.encoder is not None:
            os.makedirs(encoder_dir, exist_ok=True)

        start_time = time.time()
        # Kinda need to clean this up...
        n_envs = self.env.num_envs
        episode_rewards = np.zeros(n_envs)
        episode_returns = []
        episode_count = 0
        env_episode_ids = np.arange(n_envs, dtype=np.int32)
        env_episode_timesteps = np.zeros(n_envs, dtype=np.int32)
        next_episode_id = n_envs
        tau = 0.005

        # Episode data logging
        success_history = deque(maxlen=100)
        metric_histories = {}
        critic_loss_history = deque(maxlen=100)
        actor_loss_history = deque(maxlen=100)
        alpha_loss_history = deque(maxlen=100)

        # Local Replay buffer
        local_replay_buffer = [[] for _ in range(n_envs)]

        # In training loop, I repeatedly fuse observations since I store the raw observations into the replay buffer
        # Need to research better way
        raw_obs, _ = self.env.reset()
        with torch.no_grad():
            fused_obs = self.fuse_observations(raw_obs, detach_encoder=True)

        try: 
            for global_step in range(start_timestep, start_timestep + self.total_timesteps):
                # Get next action for current state in the episode
                total_env_steps = global_step * n_envs

                # Adding random warmup to add diverse trajectories to replay buffer
                if total_env_steps < self.random_explore_steps:
                    low = self.env.single_action_space.low
                    high = self.env.single_action_space.high
                    actions_np = np.random.uniform(low=low, high=high, size=(n_envs, low.shape[0])).astype(np.float32)
                else:
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

                # done if either terminated or truncated
                dones = np.logical_or(terminated, truncated)
                terminated_mask = terminated.astype(np.float32) 
                episode_rewards += reward

                # Info for each timestep
                step_success = np.full(n_envs, np.nan, dtype=np.float32)
                step_metrics = {}

                def _set_metric_array(name, values):
                    arr = np.asarray(values, dtype=np.float32)
                    if arr.ndim == 0:
                        arr = np.full(n_envs, float(arr), dtype=np.float32)
                    step_metrics[name] = arr

                if "success" in infos:
                    step_success = np.asarray(infos["success"], dtype=np.float32)

                if "metrics" in infos and isinstance(infos["metrics"], dict):
                    for name, values in infos["metrics"].items():
                        _set_metric_array(name, values)
                elif "distances" in infos and isinstance(infos["distances"], dict):
                    for name, values in infos["distances"].items():
                        _set_metric_array(name, values)

                if "final_info" in infos and "_final_info" in infos:
                    final_infos = infos["final_info"]
                    final_mask = infos["_final_info"]
                    for i in range(n_envs):
                        if final_mask[i] and final_infos[i] is not None:
                            final_info = final_infos[i]

                            if "success" in final_info:
                                step_success[i] = 1.0 if bool(final_info["success"]) else 0.0

                            if "metrics" in final_info and isinstance(final_info["metrics"], dict):
                                for name, value in final_info["metrics"].items():
                                    if name not in step_metrics:
                                        step_metrics[name] = np.full(n_envs, np.nan, dtype=np.float32)
                                    step_metrics[name][i] = float(value)
                            elif "distances" in final_info and isinstance(final_info["distances"], dict):
                                for name, value in final_info["distances"].items():
                                    if name not in step_metrics:
                                        step_metrics[name] = np.full(n_envs, np.nan, dtype=np.float32)
                                    step_metrics[name][i] = float(value)

                # Get states from all vectorised environments and append to replay buffer
                for i in range(n_envs):
                    obs_joints = raw_obs["joints"][i].astype(np.float32, copy=True)
                    next_obs_joints = raw_next_obs["joints"][i].astype(np.float32, copy=True)

                    if "final_observation" in infos and infos.get("_final_observation", [False] * n_envs)[i]:
                        final_obs = infos["final_observation"][i]
                        next_obs_joints = final_obs["joints"].astype(np.float32, copy=True)
                    step = StepInfo(obs_raw=obs_joints.copy(), action=actions_np[i], reward=reward[i], next_obs_raw=next_obs_joints, done=terminated_mask[i]) # Only storing if episode terminated
                    local_replay_buffer[i].append(step)

                env_episode_timesteps += 1

                for i in range(n_envs):
                    if dones[i]:
                        self.replay_buffer.add_episode(local_replay_buffer[i])
                        local_replay_buffer[i] = []

                        ret = float(episode_rewards[i])
                        episode_returns.append(ret)
                        episode_count += 1
                        episode_rewards[i] = 0.0

                        success_i = step_success[i]
                        if np.isnan(success_i):
                            success_i = 1.0 if terminated[i] else 0.0
                        success_history.append(float(success_i))

                        for metric_name, metric_values in step_metrics.items():
                            metric_val = float(metric_values[i])
                            if np.isfinite(metric_val):
                                if metric_name not in metric_histories:
                                    metric_histories[metric_name] = deque(maxlen=100)
                                metric_histories[metric_name].append(metric_val)

                        ret10 = float(np.mean(episode_returns[-10:])) if len(episode_returns) >= 1 else float("nan")
                        succ100 = float(np.mean(success_history)) if len(success_history) > 0 else float("nan")

                        metric_parts = []
                        for metric_name in sorted(metric_histories.keys()):
                            hist = metric_histories[metric_name]
                            if len(hist) > 0:
                                metric_parts.append(f"{metric_name}={np.mean(hist):.3f}")
                        metrics_summary = " | ".join(metric_parts) if metric_parts else "n/a"

                        total_env_steps = (global_step + 1) * n_envs
                        elapsed_seconds = max(1, int(time.time() - start_time))
                        formatted_time = str(datetime.timedelta(seconds=elapsed_seconds))
                        sps = int(total_env_steps / elapsed_seconds)

                        c_loss_avg = float(np.mean(critic_loss_history)) if len(critic_loss_history) > 0 else float("nan")
                        a_loss_avg = float(np.mean(actor_loss_history)) if len(actor_loss_history) > 0 else float("nan")
                        alpha_val = float(self.log_alpha.exp().item()) # Current temperature

                        if (episode_count % self.log_every_episodes == 0) or (success_i > 0.5):
                            print(
                                f"[{formatted_time}] Step: {total_env_steps} | SPS: {sps} | Ep: {episode_count} | Stage: {self.task_stage}\n"
                                f"    ├─ Returns: Current={ret:.2f} | Avg(10)={ret10:.2f} | Succ(100)={succ100:.2f}\n"
                                f"    ├─ Metrics: {metrics_summary}\n"
                                f"    └─ Network: C_Loss={c_loss_avg:.3f} | A_Loss={a_loss_avg:.3f} | Alpha={alpha_val:.4f}\n"
                                f"    └─ Device: {self.device}\n"
                                f"{'-'*75}"
                            )

                # Training
                if (self.replay_buffer.get_total_timesteps() > self.learning_steps and (global_step * n_envs) >= self.random_explore_steps):
                    for _ in range(self.updates_per_step):
                        batch = self.sample()
                        if batch is None:
                            break

                        obs_joints_, actions_, rewards_, next_obs_joints_, dones_ = batch

                        obs_batch = {"joints": obs_joints_}
                        next_obs_batch = {"joints": next_obs_joints_}

                        # Convert numpy values to tensors
                        tensor_obs = self.fuse_observations(obs_batch, detach_encoder=False)
                        tensor_next_obs = self.fuse_observations(next_obs_batch, detach_encoder=False)
                        tensor_actions = torch.as_tensor(actions_, dtype=torch.float32, device=self.device)
                        tensor_reward = torch.as_tensor(rewards_, dtype=torch.float32, device=self.device).view(-1, 1)
                        tensor_dones = torch.as_tensor(dones_, dtype=torch.float32, device=self.device).view(-1, 1)


                        # Training Critic
                        critic_loss = self.update_critic(tensor_next_obs, tensor_obs, tensor_actions, tensor_reward, tensor_dones)

                        # Train Actor: need to compare to newly updated critic
                        actor_loss, alpha_loss = self.update_actor(tensor_obs.detach())

                        critic_loss_history.append(critic_loss)
                        actor_loss_history.append(actor_loss)
                        alpha_loss_history.append(alpha_loss)

                        # Soft update the TargetCritic 
                        for target_param, local_param in zip(self.TargetCritic.parameters(), self.Critic.parameters()):
                            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                
                raw_obs = raw_next_obs
                fused_obs = fused_next_obs.detach()

                # Save model after certain amout of timesteps
                # if global_step % save_timesteps == 0 or (len(episode_returns) > 10 and np.mean(episode_returns[-10:]) >= 0):
                if global_step % save_timesteps == 0:
                    torch.save(self.Actor.state_dict(), f"{actor_dir}/{global_step}")
                    torch.save(self.Critic.state_dict(), f"{critic_dir}/{global_step}")
                    if self.encoder is not None:
                        torch.save(self.encoder.state_dict(), f"{encoder_dir}/{global_step}")
                    torch.save(
                        {
                            "log_alpha": self.log_alpha.detach().cpu(),
                            "alpha_optim": self.alpha_optim.state_dict(),
                        },
                        f"{alpha_dir}/{global_step}",
                    )
                if len(success_history) == success_history.maxlen and float(np.mean(success_history)) >= target_success_rate:
                    torch.save(self.Actor.state_dict(), f"{actor_dir}/{global_step}_solved")
                    torch.save(self.Critic.state_dict(), f"{critic_dir}/{global_step}_solved")
                    if self.encoder is not None:
                        torch.save(self.encoder.state_dict(), f"{encoder_dir}/{global_step}_solved")
                    torch.save(
                        {
                            "log_alpha": self.log_alpha.detach().cpu(),
                            "alpha_optim": self.alpha_optim.state_dict(),
                        },
                        f"{alpha_dir}/{global_step}_solved",
                    )
                    print(f"target success achieved: {np.mean(success_history):.3f}")
                    task_solved = True
                    break
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current state...")
            # Save with the current global_step or a specific "interrupted" suffix
            torch.save(self.Actor.state_dict(), f"{actor_dir}/{global_step}_interrupted")
            torch.save(self.Critic.state_dict(), f"{critic_dir}/{global_step}_interrupted")

            if self.encoder is not None:
                torch.save(self.encoder.state_dict(), f"{encoder_dir}/{global_step}_interrupted")
            torch.save(
                {
                    "log_alpha": self.log_alpha.detach().cpu(),
                    "alpha_optim": self.alpha_optim.state_dict(),
                },
                f"{alpha_dir}/{global_step}_interrupted",
            )
            print(f"Models saved at step {global_step}. Exiting.")

        return task_solved

class GlobalEpisodicReplayBuffer:
    def __init__(self, max_transitions):
        self.max_transitions = int(max_transitions)
        self.buffer = deque()
        self.episode_lengths = deque()
        self.total_transitions = 0

    def add_episode(self, episode_steps):
        if not episode_steps:
            return

        ep_len = len(episode_steps)
        self.buffer.append(episode_steps)
        self.episode_lengths.append(ep_len)
        self.total_transitions += ep_len

        while self.total_transitions > self.max_transitions and len(self.buffer) > 0:
            old_len = self.episode_lengths.popleft()
            self.buffer.popleft()
            self.total_transitions -= old_len

    def get_total_episodes(self):
        return len(self.buffer)

    def get_total_timesteps(self):
        return int(self.total_transitions)