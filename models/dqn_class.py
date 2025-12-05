import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Simple MLP used for the DQN value function.

    state_dim: dimension of observation vector
    action_dim: number of discrete actions
    hidden_dims: sizes of hidden layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers = []
        last_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple replay buffer with uniform random sampling."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        assert len(self.buffer) >= batch_size
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            t = self.buffer[idx]
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)
            next_states.append(t.next_state)
            dones.append(t.done)

        return Transition(
            state=np.stack(states),
            action=np.array(actions, dtype=np.int64),
            reward=np.array(rewards, dtype=np.float32),
            next_state=np.stack(next_states),
            done=np.array(dones, dtype=np.float32),
        )


class DQNAgent:
    """
    Minimal Double-DQN agent with a target network.

    Use:
        agent = DQNAgent(state_dim, action_dim)
        action = agent.select_action(state, epsilon)
        agent.push_transition(...)
        loss = agent.optimize_model()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        min_buffer_size: int = 1_000,
        target_update_freq: int = 1_000,
        device: Optional[torch.device] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.train_steps = 0

    # -------------------------
    # Acting
    # -------------------------
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy action selection.

        state: 1D numpy array (observation)
        epsilon: exploration rate in [0,1]
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # -------------------------
    # Experience
    # -------------------------
    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    # -------------------------
    # Learning
    # -------------------------
    def _compute_loss(self, batch: Transition) -> torch.Tensor:
        states = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Double-DQN target:
            #   a* = argmax_a Q_policy(next_state, a)
            #   target = r + gamma * Q_target(next_state, a*) * (1 - done)
            next_q_policy = self.policy_net(next_states)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q_target * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)
        return loss

    def optimize_model(self) -> Optional[float]:
        """
        Perform one gradient step if there is enough data in the buffer.
        Returns the loss value or None if no update was performed.
        """
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        self.policy_net.train()
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------------------------
    # Checkpointing
    # -------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, map_location: Optional[torch.device] = None) -> None:
        # Explicitly set weights_only=False for compatibility with PyTorch 2.6+
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device,
            weights_only=False,
        )

        # --- Case 1: New format (current save()) ---
        if "policy_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            # if no separate target state dict, just mirror policy
            target_sd = checkpoint.get("target_state_dict")
            if target_sd is not None:
                self.target_net.load_state_dict(target_sd)
            else:
                self.update_target_network()

            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)

        # --- Case 2: Older format with explicit q_network_* keys ---
        elif "q_network_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["q_network_state_dict"])
            if "target_q_network_state_dict" in checkpoint:
                self.target_net.load_state_dict(
                    checkpoint["target_q_network_state_dict"]
                )
            else:
                self.update_target_network()

            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)

        # --- Case 3: Old minimal format: just a single state_dict + metadata ---
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            hidden_sizes = checkpoint.get("hidden_sizes", (128, 128))
            state_dim = checkpoint.get("state_dim", self.state_dim)
            action_dim = checkpoint.get("action_dim", self.action_dim)

            # Update agent's idea of dims
            self.state_dim = state_dim
            self.action_dim = action_dim

            # Rebuild networks to match the checkpoint architecture
            self.policy_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)

            # Fresh optimizer tied to the new policy_net params
            # (optimizer state wasn't stored in this old format anyway)
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 1e-3
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

            self.policy_net.load_state_dict(state_dict)
            self.update_target_network()

        else:
            # If we ever hit a new weird format, this will tell us the keys.
            raise KeyError(
                f"Unrecognized checkpoint format. Available keys: {list(checkpoint.keys())}"
            )

        # Make sure target net is synced & in eval mode
        self.update_target_network()
        self.target_net.eval()
