import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


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

        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_sd_layer = nn.Linear(hidden, act_dim)

    def forward(self, x):
        x = self.network(x)
        mean = self.mean_layer(x)
        log_sd = self.log_sd_layer(x)
        log_sd = torch.clamp(log_sd, min=-20, max=2)
        return mean, log_sd


class BC():
    def __init__(self, env, device, epochs, batch_size, normalise_obs=True, lr=3e-4, seed=0):
        self.env = env
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.actor_learning_rate = float(lr)
        self.normalise_obs = normalise_obs
        self.seed = seed

        obs_space = env.observation_space
        self.observation_dim = obs_space["observation"].shape[0]
        self.goal_dim = obs_space["desired_goal"].shape[0]
        self.fused_obs_dim = self.observation_dim + self.goal_dim
        self.act_dim = env.action_space.shape[0]

        self.obs_normaliser = RunningMeanStd(self.fused_obs_dim)

        self.Actor = ActorNetwork(self.fused_obs_dim, self.act_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=self.actor_learning_rate)
        self.actor_loss = torch.nn.MSELoss()

    def load_data(self, data):
        # Split via episode instead of steps otherwise can have train test leak
        with np.load(data) as data:
            episode_index = data['episode_index']
            observation = data['observation']
            desired_goal = data['desired_goal']
            # tanh(mean) can never reach exactly +-1, so pull the saturated expert
            # actions just inside the range or the loss floors out with no gradient
            actions = np.clip(data['actions'], -0.999, 0.999)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=self.seed)
        train_index, test_index = next(gss.split(X=np.arange(len(episode_index)), groups=episode_index))

        train_input = np.concatenate([observation[train_index], desired_goal[train_index]], axis=-1)
        test_input = np.concatenate([observation[test_index], desired_goal[test_index]], axis=-1)
        train_actions = actions[train_index]
        test_actions = actions[test_index]

        if self.normalise_obs:
            # Fit on train only so the test split stays held out
            self.obs_normaliser.update(train_input)
            train_input = self.obs_normaliser.normalise(train_input)
            test_input = self.obs_normaliser.normalise(test_input)

        train_ds = TensorDataset(torch.tensor(train_input, dtype=torch.float32), torch.tensor(train_actions, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(test_input, dtype=torch.float32), torch.tensor(test_actions, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        print(f"train transitions: {len(train_ds)} | test transitions: {len(test_ds)} "
              f"| episodes: {len(np.unique(episode_index))}")

        return train_loader, test_loader

    def save_checkpoint(self, model_path, tag):
        # Same layout as the SAC agent so eval_panda.py can find both
        actor_dir = os.path.join(model_path, "Actor")
        norm_dir = os.path.join(model_path, "normaliser")
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)
        torch.save(self.Actor.state_dict(), f"{actor_dir}/{tag}")
        if self.normalise_obs:
            torch.save(self.obs_normaliser.state_dict(), f"{norm_dir}/{tag}")

    def act(self, obs):
        # Deterministic action for a single env observation, matches eval_panda.py
        fused = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        if self.normalise_obs:
            fused = self.obs_normaliser.normalise(fused)
        with torch.inference_mode():
            mean, log_sd = self.Actor(torch.as_tensor(fused, device=self.device).unsqueeze(0))
            return torch.tanh(mean).squeeze(0).cpu().numpy().astype(np.float32)

    def evaluate(self, env, episodes=20):
        # Held out loss only says how well we copy the expert, this says whether the task gets solved
        self.Actor.eval()
        successes = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=self.seed + episode)
            done = False
            success = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(self.act(obs))
                success = success or bool(info.get("is_success", False))
                done = bool(terminated or truncated)
            successes.append(float(success))
        return float(np.mean(successes))

    def train(self, data, model_path, save_epochs=5, eval_env=None, eval_episodes=20):
        train_loader, test_loader = self.load_data(data)
        best_test_loss = float("inf")

        for epoch in range(self.epochs):
            self.Actor.train()
            print(f"Epoch: {epoch}\n-------")
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                mean, log_sd = self.Actor(X)
                action = torch.tanh(mean)  # deterministic and no sampling since we're imitating
                loss = self.actor_loss(action, y)
                train_loss += loss.item()

                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

            train_loss /= len(train_loader)

            test_loss = 0

            self.Actor.eval()
            with torch.inference_mode():
                for X, y in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    mean, log_sd = self.Actor(X)
                    action = torch.tanh(mean)
                    test_loss += self.actor_loss(action, y).item()

                test_loss /= len(test_loader)

            ## Print out what's happening
            print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_checkpoint(model_path, f"{epoch}_best")

            if (epoch + 1) % save_epochs == 0:
                self.save_checkpoint(model_path, epoch)
                if eval_env is not None:
                    print(f"Success rate: {self.evaluate(eval_env, eval_episodes):.2%}")

        self.save_checkpoint(model_path, f"{self.epochs}_final")
        if eval_env is not None:
            print(f"\nFinal success rate: {self.evaluate(eval_env, eval_episodes):.2%} over {eval_episodes} episodes")
