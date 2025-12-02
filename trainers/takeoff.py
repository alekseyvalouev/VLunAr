import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from models.dqn_class import DQNAgent

TASK_NAME = "takeoff"


def linear_epsilon_decay(
    episode: int,
    max_episodes: int,
    start_eps: float = 1.0,
    end_eps: float = 0.05,
) -> float:
    frac = min(1.0, episode / max_episodes)
    return start_eps + frac * (end_eps - start_eps)


def make_env(render_mode: Optional[str] = None):
    return gym.make("LunarLander-v3", render_mode=render_mode)


def train_takeoff(
    episodes: int = 1000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 100_000,
    min_buffer_size: int = 1_000,
    target_update_freq: int = 1_000,
    model_dir: str = "checkpoints/takeoff",
) -> None:
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        buffer_size=buffer_size,
        min_buffer_size=min_buffer_size,
        target_update_freq=target_update_freq,
    )

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    best_reward = -np.inf

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        epsilon = linear_epsilon_decay(ep, episodes)

        for t in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Example takeoff shaping (optional):
            # state[1] is vertical position; you might reward gaining height.
            # shaped_reward = reward + 0.05 * next_state[1]
            shaped_reward = reward

            agent.push_transition(state, action, shaped_reward, next_state, done)
            loss = agent.optimize_model()

            state = next_state
            ep_reward += reward

            if done:
                break

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(str(model_path / "best.pt"))

        if ep % 10 == 0:
            print(
                f"[{TASK_NAME}] Episode {ep}/{episodes} | "
                f"Return: {ep_reward:.1f} | Best: {best_reward:.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()
    agent.save(str(model_path / "final.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for the takeoff task."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-buffer-size", type=int, default=1_000)
    parser.add_argument("--target-update-freq", type=int, default=1_000)
    parser.add_argument("--model-dir", type=str, default="checkpoints/takeoff")
    args = parser.parse_args()

    train_takeoff(
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        target_update_freq=args.target_update_freq,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
