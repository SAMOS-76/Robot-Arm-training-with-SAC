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

# [PANDA] goal fields stored separately instead of being sliced out of a 39-vector.
StepInfo = namedtuple(
    "StepInfo",
    ["observation", "action", "reward", "next_observation", "next_achieved_goal", "desired_goal", "done"],
)


# ----- SAC networks (copied verbatim from SAC_agent_HER.py) -----------------
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
    def __init__(self, env, device, reward_env, timesteps=1000000, replay_size=1_000_000,
                 batch_size=512, updates_per_step=2):
        self.env = env
        self.device = device
        self.total_timesteps = timesteps

        # [PANDA] dedicated un-vectorized reference env supplying the sparse reward
        # for HER relabels. We never reimplement the reward; we call its
        # compute_reward on the whole minibatch at once (vectorized numpy).
        self.reward_env = reward_env.unwrapped if hasattr(reward_env, "unwrapped") else reward_env
        self.reward_fn = self.reward_env.compute_reward

        # Hyperparameters (unchanged from SAC_agent_HER.py)
        self.critic_learning_rate = 0.0001
        self.actor_learning_rate = 0.0001
        self.gamma = 0.995
        self.batch_size = int(batch_size)
        self.replay_size = int(replay_size)
        self.updates_per_step = max(1, int(updates_per_step))
        self.learning_steps = 50_000
        self.random_explore_steps = 50_000

        # Transition-capped episodic replay (HER-compatible)
        self.replay_buffer = GlobalEpisodicReplayBuffer(max_transitions=self.replay_size)

        # [PANDA] policy input = concat(observation, desired_goal); achieved_goal excluded.
        obs_space = env.single_observation_space
        self.observation_dim = obs_space["observation"].shape[0]
        self.goal_dim = obs_space["desired_goal"].shape[0]
        self.fused_obs_dim = self.observation_dim + self.goal_dim
        act_dim = env.single_action_space.shape[0]
        self.act_dim = act_dim

        # Network Initialisation
        self.Critic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)
        self.Actor = ActorNetwork(self.fused_obs_dim, act_dim).to(self.device)
        self.TargetCritic = CriticNetworks(self.fused_obs_dim, act_dim).to(self.device)

        self.TargetCritic.load_state_dict(self.Critic.state_dict())
        for parameter in self.TargetCritic.parameters():
            parameter.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optim = torch.optim.Adam(self.Critic.parameters(), lr=self.critic_learning_rate)

        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0003)
        # [PANDA] act_dim 6 -> 4 (env-forced), so target_entropy moves -6 -> -4.
        self.target_entropy = -act_dim

        self.log_every_episodes = 10

    # [PANDA] policy/critic input builder: concat(observation, desired_goal).
    def fuse_observations(self, obs):
        observation = torch.as_tensor(np.asarray(obs["observation"]), dtype=torch.float32, device=self.device)
        desired_goal = torch.as_tensor(np.asarray(obs["desired_goal"]), dtype=torch.float32, device=self.device)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
            desired_goal = desired_goal.unsqueeze(0)
        return torch.cat([observation, desired_goal], dim=-1)

    # ----- SAC updates (copied verbatim from SAC_agent_HER.py) ---------------
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

    # [PANDA] HER relabel: SAME strategy as SAC_agent_HER.sample() (80% future
    # relabel, episode sampled weighted by length, future-only goal), but the goal
    # is read from dict-key fields instead of array slices, the relabeled reward
    # comes from the reference env's compute_reward (batched), and done stays 0.
    def sample(self):
        if self.replay_buffer.get_total_episodes() == 0:
            return None
        if self.replay_buffer.get_total_timesteps() < self.batch_size:
            return None

        episodes = list(self.replay_buffer.buffer)
        lengths = np.asarray(self.replay_buffer.episode_lengths, dtype=np.float64)
        probs = lengths / lengths.sum()
        episode_indices = np.random.choice(len(episodes), size=self.batch_size, replace=True, p=probs)  # Weighted sampling to prioritise sampling from longer eps
        HER_masking = np.random.rand(self.batch_size) < 0.8

        obs_inputs = np.empty((self.batch_size, self.fused_obs_dim), dtype=np.float32)
        next_obs_inputs = np.empty((self.batch_size, self.fused_obs_dim), dtype=np.float32)
        actions = np.empty((self.batch_size, self.act_dim), dtype=np.float32)
        rewards = np.empty((self.batch_size,), dtype=np.float32)
        dones = np.zeros((self.batch_size,), dtype=np.float32)  # [PANDA] fully non-terminal

        # Collect relabeled transitions to compute their reward in one batched call.
        relabel_indices = []
        relabel_next_ag = []
        relabel_new_goal = []

        for index, episode_id in enumerate(episode_indices):
            episode = episodes[episode_id]
            length = len(episode)
            relabel = bool(HER_masking[index] and length > 1)
            if relabel:
                step = np.random.randint(0, length - 1)  # guarantee a future step exists
            else:
                step = np.random.randint(0, length)
            timestep = episode[step]

            observation = np.asarray(timestep.observation, dtype=np.float32)
            next_observation = np.asarray(timestep.next_observation, dtype=np.float32)
            actions[index] = np.asarray(timestep.action, dtype=np.float32)

            if relabel:
                future_timestep = np.random.randint(step + 1, length)
                new_goal = np.asarray(episode[future_timestep].next_achieved_goal, dtype=np.float32)  # future achieved goal
                goal = new_goal
                # Reward recomputed below via reference env compute_reward (batched).
                relabel_indices.append(index)
                relabel_next_ag.append(np.asarray(timestep.next_achieved_goal, dtype=np.float32))
                relabel_new_goal.append(new_goal)
                rewards[index] = 0.0     # placeholder, overwritten after the batched call
                dones[index] = 0.0       # [PANDA] success no longer forces done=1
            else:
                goal = np.asarray(timestep.desired_goal, dtype=np.float32)
                rewards[index] = np.float32(timestep.reward)
                dones[index] = np.float32(timestep.done)

            obs_inputs[index] = np.concatenate([observation, goal])
            next_obs_inputs[index] = np.concatenate([next_observation, goal])

        # [PANDA] one vectorized compute_reward over the whole relabeled subset.
        if relabel_indices:
            ag_batch = np.stack(relabel_next_ag, axis=0)
            dg_batch = np.stack(relabel_new_goal, axis=0)
            relabel_rewards = np.asarray(
                self.reward_fn(ag_batch, dg_batch, [{} for _ in relabel_indices]),
                dtype=np.float32,
            ).reshape(-1)
            for j, index in enumerate(relabel_indices):
                rewards[index] = relabel_rewards[j]

        return obs_inputs, actions, rewards, next_obs_inputs, dones

    # Modified existing load checkpoint code (encoder path removed for panda)
    def load_checkpoint(self, model_path, timestep, load_critic=True):
        actor_path = os.path.join(model_path, "Actor", str(timestep))
        critic_path = os.path.join(model_path, "Critic", str(timestep))
        alpha_path = os.path.join(model_path, "Alpha", str(timestep))

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

        print(f"Loaded checkpoint at timestep {timestep}")

    def train(self, model_path, save_timesteps=50000, start_timestep=0, debug_boundary=False):
        actor_dir = os.path.join(model_path, "Actor")
        critic_dir = os.path.join(model_path, "Critic")
        alpha_dir = os.path.join(model_path, "Alpha")
        os.makedirs(alpha_dir, exist_ok=True)
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(critic_dir, exist_ok=True)

        start_time = time.time()
        n_envs = self.env.num_envs
        episode_rewards = np.zeros(n_envs)
        episode_returns = []
        episode_count = 0
        tau = 0.005

        # [PANDA] NEXT_STEP autoreset bookkeeping: when an env is done at step t,
        # the NEXT step is a throwaway reset step (action ignored, reward is a
        # placeholder). We must not store that transition.
        autoreset_pending = np.zeros(n_envs, dtype=bool)

        # Episode data logging
        success_history = deque(maxlen=100)
        critic_loss_history = deque(maxlen=100)
        actor_loss_history = deque(maxlen=100)
        alpha_loss_history = deque(maxlen=100)

        # Replay buffer (per-env in-progress episodes; flushed on done)
        local_replay_buffer = [[] for _ in range(n_envs)]

        raw_obs, _ = self.env.reset()
        with torch.no_grad():
            fused_obs = self.fuse_observations(raw_obs)

        try:
            for global_step in range(start_timestep, start_timestep + self.total_timesteps):
                total_env_steps = global_step * n_envs

                # Random warmup to seed the replay buffer with diverse trajectories.
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

                # [PANDA] === TEMP BOUNDARY DEBUG (remove) ===
                if debug_boundary and global_step == start_timestep:
                    print("=" * 30 + " TEMP BOUNDARY DEBUG (remove) " + "=" * 30)
                    print("obs dict keys:", list(raw_next_obs.keys()))
                    for _k, _v in raw_next_obs.items():
                        _a = np.asarray(_v)
                        print(f"  obs['{_k}']: shape={_a.shape} dtype={_a.dtype}")
                    print("action batch shape:", np.asarray(actions_np).shape, "dtype:", np.asarray(actions_np).dtype)
                    print("reward shape:", np.asarray(reward).shape, "dtype:", np.asarray(reward).dtype)
                    print("info keys:", list(infos.keys()))
                    print("fused_obs_dim:", self.fused_obs_dim, "(= observation", self.observation_dim, "+ goal", self.goal_dim, ")")
                    print("act_dim:", self.act_dim, "target_entropy:", self.target_entropy)
                    _ag = np.asarray(raw_next_obs["achieved_goal"])
                    _dg = np.asarray(raw_next_obs["desired_goal"])
                    print("compute_reward(ag, dg) sample[:3]:", np.asarray(self.reward_fn(_ag, _dg, [{} for _ in range(len(_ag))]))[:3])
                    print("compute_reward(ag, ag) sample[:3] (expect ~0):", np.asarray(self.reward_fn(_ag, _ag, [{} for _ in range(len(_ag))]))[:3])
                    print("=" * 90)
                # [PANDA] === END TEMP BOUNDARY DEBUG ===

                dones = np.logical_or(terminated, truncated)
                is_autoreset = autoreset_pending.copy()      # envs being reset THIS step
                autoreset_pending[:] = False

                # Mask placeholder reward on autoreset steps before accounting.
                episode_rewards += np.where(is_autoreset, 0.0, reward)

                # [PANDA] success comes from info["is_success"] (+ "_is_success" mask).
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

                        if (episode_count % self.log_every_episodes == 0) or (success_i > 0.5):
                            print(
                                f"[{formatted_time}] Step: {total_env_steps} | SPS: {sps} | Ep: {episode_count}\n"
                                f"    ├─ Returns: Current={ret:.2f} | Avg(10)={ret10:.2f} | Succ(100)={succ100:.2f}\n"
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

                        obs_inputs_, actions_, rewards_, next_obs_inputs_, dones_ = batch

                        # [PANDA] sample() already returns concat(observation, goal); tensorize directly.
                        tensor_obs = torch.as_tensor(obs_inputs_, dtype=torch.float32, device=self.device)
                        tensor_next_obs = torch.as_tensor(next_obs_inputs_, dtype=torch.float32, device=self.device)
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
                fused_obs = fused_next_obs = self.fuse_observations(raw_next_obs)

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
            print(f"Models saved at step {global_step}. Exiting.")


# Transition-capped episodic replay buffer (copied verbatim from SAC_agent_HER.py)
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
