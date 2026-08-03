"""
SAC + HER agent adapted to panda-gym's `PandaPickAndPlace-v3` (a Gymnasium goal env).

This is an INTERFACE port of SAC_agent_HER.py. The SAC update rules
(`update_critic`, `update_actor`, alpha tuning) and the HER *strategy*
(80% future-relabel, episode-length-weighted sampling, transition-capped
episodic buffer) are copied unchanged from SAC_agent_HER.py. Only the code that
touches the environment differs, and every such difference is marked with a
`# [PANDA]` comment so it can be diffed against the original:

  * observation is now Dict{observation, achieved_goal, desired_goal}; the policy
    input is concat(observation, desired_goal). `achieved_goal` is used ONLY for
    HER relabeling + reward, never fed to a network.
  * reward for relabeled transitions comes from a dedicated reference env's
    `compute_reward` (vectorized over the minibatch) -- the sparse reward is NOT
    reimplemented here.
  * episodes are fully NON-TERMINAL: panda-gym `terminated` is always False and
    episodes end by truncation at 50 steps, so stored `done` is always 0.
  * the env uses Gymnasium 1.x `AutoresetMode.NEXT_STEP`: the terminal obs is
    returned inline on the truncating step, and the FOLLOWING step is a throwaway
    reset step (ignored action / placeholder reward) that must be skipped. There
    is no `final_observation` info key under this autoreset mode.

The original camera/CNN-encoder path is intentionally dropped (panda obs has no
image). SAC_agent_HER.py and the custom stacking env are left untouched.
"""
import os
import datetime
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from collections import deque, namedtuple

# Data struct for timestep info
StepInfo = namedtuple(
    "StepInfo",
    ["observation", "action", "reward", "next_observation", "next_achieved_goal", "desired_goal", "done"],
)


class RunningMeanStd:
    # Normalise the observations so they are around the same range
    # Ensures some observations don't dominate over small observations

    def __init__(self, dim, eps=1e-4):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = float(eps)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total
        self.var = M2 / total
        self.count = total

    def normalise(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5.0, 5.0).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


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

        # Separating outputs for mean and log_sd 
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_sd_layer = nn.Linear(hidden, act_dim)

    def forward(self, x):
        x = self.network(x)
        mean = self.mean_layer(x)
        log_sd = self.log_sd_layer(x)
        # clamp so values aren't too large or small giving infinity or NaN
        log_sd = torch.clamp(log_sd, min=-20, max=2)
        return mean, log_sd


class CriticNetworks(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        self.network1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
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
            nn.Linear(hidden, 1)
        )

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        q1 = self.network1(x)
        q2 = self.network2(x)
        return q1, q2


# SAC class
class SACAgent():
    def __init__(self, env, device, reward_env, timesteps=1000000, replay_size=1_000_000, batch_size=512, updates_per_step=8,
                 random_explore_steps=1_000, learning_steps=1_000, her_ratio=0.8, lr=3e-4, normalise_obs=True):
        self.env = env
        self.device = device
        self.total_timesteps = timesteps

        # Get same reward environment and function from the PandaGym environment
        # Easier to do like this then rather trying to copy my custom reward as this is already layed out for me
        self.reward_env = reward_env.unwrapped if hasattr(reward_env, "unwrapped") else reward_env
        self.reward_fn = self.reward_env.compute_reward
        self.critic_learning_rate = float(lr)
        self.actor_learning_rate = float(lr)
        # Retuned gamma from 0.995 to 0.95 as found that it was prioritising later rewards too much
        self.gamma = 0.95
        self.batch_size = int(batch_size)
        self.replay_size = int(replay_size)
        self.updates_per_step = int(updates_per_step) # What is updates_per_step doin again
        self.her_ratio = float(her_ratio)
        self.learning_steps = int(learning_steps)              # min buffer transitions before any gradient update
        self.random_explore_steps = int(random_explore_steps)  # env steps of uniform-random actions before using the actor

        # Note for panda achieved goal is a part of observation when getting the observation space
        obs_space = env.single_observation_space
        self.observation_dim = obs_space["observation"].shape[0]
        self.goal_dim = obs_space["desired_goal"].shape[0]
        self.fused_obs_dim = self.observation_dim + self.goal_dim
        act_dim = env.single_action_space.shape[0]
        self.act_dim = act_dim

        self.replay_buffer = GlobalEpisodicReplayBuffer(max_transitions=self.replay_size, obs_dim=self.observation_dim, act_dim=self.act_dim, goal_dim=self.goal_dim)

        # Observation normalisation over the network input concat(observation, desired_goal).
        # Raw achieved/desired goals are kept for HER relabel + compute_reward (real units).
        # Normalisation only for network input
        self.normalise_obs = bool(normalise_obs)
        self.obs_normaliser = RunningMeanStd(self.fused_obs_dim)

        # Network Initialisation
        self.Critic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)
        self.Actor = ActorNetwork(self.fused_obs_dim, act_dim).to(self.device)
        self.TargetCritic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)
        # Make target network weights the same as critic
        self.TargetCritic.load_state_dict(self.Critic.state_dict())
        for parameter in self.TargetCritic.parameters():
            parameter.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optim = torch.optim.Adam(self.Critic.parameters(), lr=self.critic_learning_rate)

        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0003)
        self.target_entropy = -act_dim

        self.log_every_episodes = 10

    # fuse observations and normalise into format for networks
    def fuse_observations(self, obs):
        observation = np.asarray(obs["observation"], dtype=np.float32)
        desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        if observation.ndim == 1: # Add dimentionality if only 1 env is being used like in eval
            observation = observation[None, :]
            desired_goal = desired_goal[None, :]
        fused = np.concatenate([observation, desired_goal], axis=1)
        if self.normalise_obs:
            fused = self.obs_normaliser.normalise(fused)
        return torch.as_tensor(fused, dtype=torch.float32, device=self.device)

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

        # Below was very unstable changed to alpha trick
        # alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        alpha_loss_value = alpha_loss.item()

        policy_entropy = (-log_prob).mean().item()
        return actor_loss.item(), alpha_loss_value, policy_entropy

    def sample(self):
        buf = self.replay_buffer
        if buf.size < self.batch_size:
            return None

        # With the nature of the buffer used, some spaces are invalide so have to keep track when sampling
        sampled_indices = buf.valid_indices[np.random.randint(0, buf.valid_indices.shape[0], size=self.batch_size)]
        episode_end = buf.ep_end[sampled_indices] # Get the index of the episode end for each sampled transition
        her_mask = (np.random.rand(self.batch_size) < self.her_ratio) & (sampled_indices < episode_end - 1)

        # For each HER transition, pick a random future step within the same episode.
        steps_remaining = np.maximum(episode_end - sampled_indices - 1, 1)
        future_indices = sampled_indices + 1 + (np.random.rand(self.batch_size) * steps_remaining).astype(np.int64)
        future_indices = np.minimum(future_indices, episode_end - 1)           
        future_indices = np.where(her_mask, future_indices, sampled_indices)   # non-HER: point to self (value unused)

        # Get achieved goal from next time step and relabel desired goal to that if HER index
        future_achieved_goal = buf.next_achieved_goal[future_indices]
        goals = buf.desired_goal[sampled_indices].copy()
        goals[her_mask] = future_achieved_goal[her_mask]

        # Start with stored rewards (correct for non-HER transitions).
        # For HER transitions recompute: "did this step's outcome satisfy the pretend goal?"
        rewards = buf.reward[sampled_indices].astype(np.float32, copy=True)
        if her_mask.any():
            n_her = int(her_mask.sum())
            her_rewards = np.asarray(self.reward_fn(buf.next_achieved_goal[sampled_indices][her_mask], future_achieved_goal[her_mask], [{} for _ in range(n_her)]), dtype=np.float32).reshape(-1)
            rewards[her_mask] = her_rewards

        # Panda episodes always end by truncation, never a true terminal state, so done is always 0.
        dones = np.zeros(self.batch_size, dtype=np.float32)

        # Build the final network inputs: concat(observation, goal) — same layout as fuse_observations().
        obs_inputs = np.concatenate([buf.obs[sampled_indices], goals], axis=1)
        next_obs_inputs = np.concatenate([buf.next_obs[sampled_indices], goals], axis=1)
        actions = buf.action[sampled_indices].astype(np.float32, copy=True)

        return obs_inputs, actions, rewards, next_obs_inputs, dones

    def load_checkpoint(self, model_path, timestep, load_critic=True):
        actor_path = os.path.join(model_path, "Actor", str(timestep))
        critic_path = os.path.join(model_path, "Critic", str(timestep))
        alpha_path = os.path.join(model_path, "Alpha", str(timestep))
        norm_path = os.path.join(model_path, "normaliser", str(timestep))

        if not os.path.isfile(actor_path):
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")

        actor_state = torch.load(actor_path, map_location=self.device)
        self.Actor.load_state_dict(actor_state)

        if load_critic:
            if not os.path.isfile(critic_path):
                raise FileNotFoundError(f"Critic checkpoint not found: {critic_path}")
            critic_state = torch.load(critic_path, map_location=self.device)
            self.Critic.load_state_dict(critic_state)
            self.TargetCritic.load_state_dict(self.Critic.state_dict())

        if os.path.isfile(alpha_path):
            alpha_state = torch.load(alpha_path, map_location=self.device)
            if "log_alpha" in alpha_state:
                self.log_alpha.data.copy_(alpha_state["log_alpha"].to(self.device))
            if "alpha_optim" in alpha_state:
                self.alpha_optim.load_state_dict(alpha_state["alpha_optim"])

        if self.normalise_obs and os.path.isfile(norm_path):
            self.obs_normaliser.load_state_dict(torch.load(norm_path, map_location="cpu", weights_only=False))

        print(f"Loaded checkpoint at timestep {timestep}")

    def train(self, model_path, save_timesteps=50000, start_timestep=0, debug_boundary=False):
        actor_dir = os.path.join(model_path, "Actor")
        critic_dir = os.path.join(model_path, "Critic")
        alpha_dir = os.path.join(model_path, "Alpha")
        norm_dir = os.path.join(model_path, "normaliser")
        os.makedirs(alpha_dir, exist_ok=True)
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(critic_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)

        start_time = time.time()
        n_envs = self.env.num_envs

        alpha_mode = f"auto(target_entropy={self.target_entropy})"
        print(
            f"{'='*75}\n"
            f"SAC+HER on panda-gym | device={self.device} | n_envs={n_envs}\n"
            f"    gamma={self.gamma} | lr={self.actor_learning_rate} | batch={self.batch_size} "
            f"| updates/step={self.updates_per_step} | obs_norm={self.normalise_obs}\n"
            f"    her_ratio={self.her_ratio} | alpha={alpha_mode} | warmup={self.random_explore_steps} env-steps\n"
            f"    budget={self.total_timesteps} global steps = {self.total_timesteps * n_envs} env-steps\n"
            f"{'='*75}",
            flush=True,
        )

        episode_rewards = np.zeros(n_envs)
        episode_returns = []
        episode_count = 0
        tau = 0.005

        # If an env finishes the timestep after it finishes is an inbetween step which shouldn't be sotred
        # Need to keep track of which envs have/not been resetto not add the reset timestep to the buffer
        autoreset_pending = np.zeros(n_envs, dtype=bool)

        # Episode data logging
        success_history = deque(maxlen=100)
        critic_loss_history = deque(maxlen=100)
        actor_loss_history = deque(maxlen=100)
        alpha_loss_history = deque(maxlen=100)
        entropy_history = deque(maxlen=100)
        r0_frac_history = deque(maxlen=100)

        # local replay buffer per env to keep track
        local_replay_buffer = [[] for _ in range(n_envs)]

        raw_obs, _ = self.env.reset()
        with torch.no_grad():
            fused_obs = self.fuse_observations(raw_obs)

        try:
            for global_step in range(start_timestep, start_timestep + self.total_timesteps):
                total_env_steps = global_step * n_envs

                # Random warmup
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

                dones = np.logical_or(terminated, truncated)
                is_autoreset = autoreset_pending.copy()      # envs being reset THIS step
                autoreset_pending[:] = False

                # Update obs-normalisation stats from the acted observations (network-input distribution).
                if self.normalise_obs:
                    self.obs_normaliser.update(
                        np.concatenate(
                            [np.asarray(raw_obs["observation"], dtype=np.float32),
                             np.asarray(raw_obs["desired_goal"], dtype=np.float32)],
                            axis=1,
                        )
                    )

                # Mask placeholder reward on autoreset steps before accounting.
                episode_rewards += np.where(is_autoreset, 0.0, reward)

                # In panda success comes from info["is_success"] (+ "_is_success" mask).
                step_success = np.full(n_envs, np.nan, dtype=np.float32)
                if "is_success" in infos:
                    succ_arr = np.asarray(infos["is_success"], dtype=np.float32)
                    mask = np.asarray(infos.get("_is_success", np.ones(n_envs, dtype=bool)), dtype=bool)
                    step_success[mask] = succ_arr[mask]

                # Store transitions (skip throwaway autoreset steps).
                for i in range(n_envs):
                    if is_autoreset[i]:
                        continue
                    # Under NEXT_STEP autoreset raw_next_obs is the TRUE terminal obs on a done step.
                    step = StepInfo(
                        observation=np.asarray(raw_obs["observation"][i], dtype=np.float32).copy(),
                        action=actions_np[i].copy(),
                        reward=np.float32(reward[i]),
                        next_observation=np.asarray(raw_next_obs["observation"][i], dtype=np.float32).copy(),
                        next_achieved_goal=np.asarray(raw_next_obs["achieved_goal"][i], dtype=np.float32).copy(),
                        desired_goal=np.asarray(raw_obs["desired_goal"][i], dtype=np.float32).copy(),
                        done=np.float32(0.0),  # [PANDA] fully non-terminal
                    )
                    local_replay_buffer[i].append(step)

                # Episode end handling (a done step is never an autoreset step under NEXT_STEP).
                for i in range(n_envs):
                    if dones[i] and not is_autoreset[i]:
                        self.replay_buffer.add_episode(local_replay_buffer[i])
                        local_replay_buffer[i] = []
                        autoreset_pending[i] = True  # next step for this env is a reset

                        ret = float(episode_rewards[i])
                        episode_returns.append(ret)
                        episode_count += 1
                        episode_rewards[i] = 0.0

                        success_i = step_success[i]
                        if np.isnan(success_i):
                            success_i = 0.0
                        success_history.append(float(success_i))

                        ret10 = float(np.mean(episode_returns[-10:])) if len(episode_returns) >= 1 else float("nan")
                        succ100 = float(np.mean(success_history)) if len(success_history) > 0 else float("nan")

                        total_env_steps = (global_step + 1) * n_envs
                        elapsed_seconds = max(1, int(time.time() - start_time))
                        formatted_time = str(datetime.timedelta(seconds=elapsed_seconds))
                        sps = int(total_env_steps / elapsed_seconds)

                        c_loss_avg = float(np.mean(critic_loss_history)) if len(critic_loss_history) > 0 else float("nan")
                        a_loss_avg = float(np.mean(actor_loss_history)) if len(actor_loss_history) > 0 else float("nan")
                        alpha_val = float(self.log_alpha.exp().item())

                        ent_avg = float(np.mean(entropy_history)) if len(entropy_history) > 0 else float("nan")
                        r0_avg = float(np.mean(r0_frac_history)) if len(r0_frac_history) > 0 else float("nan")

                        # Throttle to every Nth episode. (Dropped the per-success trigger, which
                        # floods the console once the policy starts succeeding across n_envs.)
                        if episode_count % self.log_every_episodes == 0:
                            print(
                                f"[{formatted_time}] Step: {total_env_steps} | SPS: {sps} | Ep: {episode_count}\n"
                                f"    ├─ Returns: Current={ret:.2f} | Avg(10)={ret10:.2f} | Succ(100)={succ100:.2f}\n"
                                f"    ├─ Network: C_Loss={c_loss_avg:.3f} | A_Loss={a_loss_avg:.3f} | Alpha={alpha_val:.4f}\n"
                                f"    └─ Diag   : Entropy={ent_avg:.2f} (target {self.target_entropy}) | Batch_r0_frac={r0_avg:.3f}\n"
                                f"{'-'*75}"
                            )

                # Training
                if (self.replay_buffer.get_total_timesteps() > self.learning_steps and (global_step * n_envs) >= self.random_explore_steps):
                    for _ in range(self.updates_per_step):
                        batch = self.sample()
                        if batch is None:
                            break

                        obs_inputs_, actions_, rewards_, next_obs_inputs_, dones_ = batch

                        # sample() returns RAW concat(observation, goal) so I normalise then convert to tensor 
                        # (same normaliser as the act-time fuse_observations, for consistency).
                        if self.normalise_obs:
                            obs_inputs_ = self.obs_normaliser.normalise(obs_inputs_)
                            next_obs_inputs_ = self.obs_normaliser.normalise(next_obs_inputs_)
                        tensor_obs = torch.as_tensor(obs_inputs_, dtype=torch.float32, device=self.device)
                        tensor_next_obs = torch.as_tensor(next_obs_inputs_, dtype=torch.float32, device=self.device)
                        tensor_actions = torch.as_tensor(actions_, dtype=torch.float32, device=self.device)
                        tensor_reward = torch.as_tensor(rewards_, dtype=torch.float32, device=self.device).view(-1, 1)
                        tensor_dones = torch.as_tensor(dones_, dtype=torch.float32, device=self.device).view(-1, 1)

                        # Training Critic
                        critic_loss = self.update_critic(tensor_next_obs, tensor_obs, tensor_actions, tensor_reward, tensor_dones)

                        # Train Actor: need to compare to newly updated critic
                        actor_loss, alpha_loss, policy_entropy = self.update_actor(tensor_obs.detach())

                        critic_loss_history.append(critic_loss)
                        actor_loss_history.append(actor_loss)
                        alpha_loss_history.append(alpha_loss)
                        entropy_history.append(policy_entropy)
                        r0_frac_history.append(float(np.mean(rewards_ > -0.5)))

                        # Soft update the TargetCritic
                        for target_param, local_param in zip(self.TargetCritic.parameters(), self.Critic.parameters()):
                            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

                raw_obs = raw_next_obs
                fused_obs = self.fuse_observations(raw_next_obs)

                # Save model after certain amount of timesteps
                if global_step % save_timesteps == 0:
                    torch.save(self.Actor.state_dict(), f"{actor_dir}/{global_step}")
                    torch.save(self.Critic.state_dict(), f"{critic_dir}/{global_step}")
                    torch.save(
                        {
                            "log_alpha": self.log_alpha.detach().cpu(),
                            "alpha_optim": self.alpha_optim.state_dict(),
                        },
                        f"{alpha_dir}/{global_step}",
                    )
                    if self.normalise_obs:
                        torch.save(self.obs_normaliser.state_dict(), f"{norm_dir}/{global_step}")
                target_success_rate = 0.80
                if len(success_history) == success_history.maxlen and float(np.mean(success_history)) >= target_success_rate:
                    torch.save(self.Actor.state_dict(), f"{actor_dir}/{global_step}_solved")
                    torch.save(self.Critic.state_dict(), f"{critic_dir}/{global_step}_solved")
                    torch.save(
                        {
                            "log_alpha": self.log_alpha.detach().cpu(),
                            "alpha_optim": self.alpha_optim.state_dict(),
                        },
                        f"{alpha_dir}/{global_step}_solved",
                    )
                    if self.normalise_obs:
                        torch.save(self.obs_normaliser.state_dict(), f"{norm_dir}/{global_step}_solved")
                    print(f"target success achieved: {np.mean(success_history):.3f}")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current state...")
            torch.save(self.Actor.state_dict(), f"{actor_dir}/{global_step}_interrupted")
            torch.save(self.Critic.state_dict(), f"{critic_dir}/{global_step}_interrupted")
            torch.save(
                {
                    "log_alpha": self.log_alpha.detach().cpu(),
                    "alpha_optim": self.alpha_optim.state_dict(),
                },
                f"{alpha_dir}/{global_step}_interrupted",
            )
            if self.normalise_obs:
                torch.save(self.obs_normaliser.state_dict(), f"{norm_dir}/{global_step}_interrupted")
            print(f"Models saved at step {global_step}. Exiting.")


class GlobalEpisodicReplayBuffer:
    def __init__(self, max_transitions, obs_dim, act_dim, goal_dim):
        self.capacity = int(max_transitions)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.next_achieved_goal = np.zeros((self.capacity, goal_dim), dtype=np.float32)
        self.desired_goal = np.zeros((self.capacity, goal_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.ep_start = np.zeros((self.capacity,), dtype=np.int64)   # per-slot: start of its episode
        self.ep_end = np.zeros((self.capacity,), dtype=np.int64)     # per-slot: exclusive end of its episode
        self.valid = np.zeros((self.capacity,), dtype=bool)
        self.write_ptr = 0
        self.size = 0
        self.valid_indices = np.empty((0,), dtype=np.int64)

    def add_episode(self, episode_steps):
        T = len(episode_steps)
        if T == 0:
            return
        if T > self.capacity:                       # pathological; keep the tail
            episode_steps = episode_steps[-self.capacity:]
            T = self.capacity

        # Keep episodes contiguous: wrap to slot 0 if it won't fit before the ring end.
        if self.write_ptr + T > self.capacity:
            self.valid[self.write_ptr:self.capacity] = False
            self.write_ptr = 0

        start = self.write_ptr
        end = start + T

        # Evict every OLD episode whose slots overlap [start, end) -- invalidate each one fully.
        i = start
        while i < end:
            if self.valid[i]:
                os_, oe_ = int(self.ep_start[i]), int(self.ep_end[i])
                self.valid[os_:oe_] = False
                i = oe_
            else:
                i += 1

        # Write the new episode contiguously.
        for k, s in enumerate(episode_steps):
            j = start + k
            self.obs[j] = s.observation
            self.next_obs[j] = s.next_observation
            self.action[j] = s.action
            self.next_achieved_goal[j] = s.next_achieved_goal
            self.desired_goal[j] = s.desired_goal
            self.reward[j] = s.reward
        self.ep_start[start:end] = start
        self.ep_end[start:end] = end
        self.valid[start:end] = True
        self.write_ptr = end if end < self.capacity else 0

        self.valid_indices = np.flatnonzero(self.valid)
        self.size = int(self.valid_indices.shape[0])

    def get_total_episodes(self):
        return 1 if self.size > 0 else 0

    def get_total_timesteps(self):
        return int(self.size)
