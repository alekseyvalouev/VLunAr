#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

from models.dqn_class import DQNAgent
from environments.custom_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    FPS,
)

TASK_NAME = "land"

# Scale to convert between world vertical velocity and the normalized state space
VY_SCALE = (VIEWPORT_H / SCALE / 2.0) / FPS

# Target *world* vertical speed for a smooth descent (negative is downward).
TARGET_WORLD_VY = -0.3  # gentle-ish descent in world units

# Convert to normalized state velocity units: state_vy = world_vy / VY_SCALE
TARGET_STATE_VY = TARGET_WORLD_VY / VY_SCALE


def linear_epsilon_decay(
    episode: int,
    max_episodes: int,
    start_eps: float = 1.0,
    end_eps: float = 0.05,
) -> float:
    frac = min(1.0, episode / max_episodes)
    return start_eps + frac * (end_eps - start_eps)


def make_env(render_mode: Optional[str] = None):
    env = LunarLander(
        render_mode=render_mode,
        continuous=False,
        random_initial_force=False,
    )
    return env


def shaped_reward_vertical_land(
    prev_state: np.ndarray,
    next_state: np.ndarray,
    start_x: float,
    env_reward: float,
    done: bool,
) -> float:
    """
    Shaping that nudges the lander to:

      - Go straight down (decreasing y, negative vy near TARGET_STATE_VY)
      - Stay near the initial x-position (vertical line)
      - Stay upright (small angle, small angular velocity)

    We *add* this on top of the base env_reward (scaled down), and keep
    magnitudes small so the agent can actually learn.
    """

    # Unpack states (assume standard LunarLander layout)
    x = float(next_state[0])
    y = float(next_state[1])
    vx = float(next_state[2])
    vy = float(next_state[3])
    angle = float(next_state[4])
    angular_vel = float(next_state[5])

    prev_y = float(prev_state[1])
    prev_vy = float(prev_state[3])

    shaping = 0.0

    # ------------------------------------------------------------------
    # 1. Stay on the starting vertical line (x ≈ start_x) and small vx
    # ------------------------------------------------------------------
    dx = x - start_x
    # Quadratic penalties but small coefficients
    shaping += -1.0 * (dx ** 2)
    shaping += -0.25 * (vx ** 2)

    # ------------------------------------------------------------------
    # 2. Encourage going DOWN and not hovering way up
    # ------------------------------------------------------------------

    # 2a. Reward reduction in height (y going towards 0)
    dy = y - prev_y
    if dy < 0.0:
        # Moving down: small positive reward
        shaping += 0.5 * (-dy)
    else:
        # Moving up: small penalty
        shaping += -0.5 * dy

    # 2b. Encourage vy close to a target downward speed
    vy_error = vy - TARGET_STATE_VY
    shaping += -0.2 * (vy_error ** 2)

    # Extra mild penalty if going upward
    if vy > 0.0:
        shaping += -0.2 * (vy ** 2)

    # Smoothness in vertical speed (avoid crazy bouncing)
    dv = vy - prev_vy
    shaping += -0.05 * (dv ** 2)

    # ------------------------------------------------------------------
    # 3. Keep the lander aligned (no tilt, low spin)
    # ------------------------------------------------------------------
    shaping += -1.5 * (angle ** 2)
    shaping += -0.3 * (angular_vel ** 2)

    # Tiny bonus for being really upright & stable
    if abs(angle) < 0.05 and abs(angular_vel) < 0.05:
        shaping += 0.5

    # ------------------------------------------------------------------
    # 4. Tiny living penalty so it doesn't just stall forever
    # ------------------------------------------------------------------
    shaping += -0.005

    # ------------------------------------------------------------------
    # 5. Let the env tell us about actual landing/crash, but shrink it
    # ------------------------------------------------------------------
    # env_reward is usually in [-100, +100] for LunarLander.
    # Scale it down so shaping dominates *direction* but env reward still
    # gives a landing signal.
    total_reward = 0.1 * env_reward + shaping

    # Clamp total reward to avoid huge spikes that destabilize learning
    total_reward = float(np.clip(total_reward, -10.0, 10.0))

    return total_reward


def train_land(
    episodes: int = 1000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 100_000,
    min_buffer_size: int = 1_000,
    target_update_freq: int = 1_000,
    model_dir: str = "checkpoints/land",
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

    model_path = Path(model_dir) / TASK_NAME
    model_path.mkdir(parents=True, exist_ok=True)
    best_score = -np.inf

    # Randomize initial (x, y); the "vertical line" is at start_x in state space
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE
    safe_min_y = world_h * 0.6
    safe_max_y = world_h * 0.9

    for ep in range(1, episodes + 1):
        init_x = np.random.uniform(0.0, world_w)
        init_y = np.random.uniform(safe_min_y, safe_max_y)
        env.unwrapped.custom_init_x = init_x
        env.unwrapped.custom_init_y = init_y
        env.unwrapped.random_initial_force = False

        state, _ = env.reset()
        done = False

        # Define the "vertical line" at the initial x position in state space
        start_x = state[0]
        ep_score = 0.0
        epsilon = linear_epsilon_decay(ep, episodes)

        for t in range(max_steps):
            prev_state = state.copy()

            # Agent picks action freely
            action = agent.select_action(state, epsilon)

            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = shaped_reward_vertical_land(
                prev_state,
                next_state,
                start_x,
                env_reward,
                done,
            )

            agent.push_transition(state, action, reward, next_state, done)
            agent.optimize_model()

            state = next_state
            ep_score += reward

            if done:
                break

        if ep_score > best_score:
            best_score = ep_score
            agent.save(str(model_path / "best.pt"))

        if ep % 10 == 0:
            print(
                f"[{TASK_NAME}] Episode {ep:04d}/{episodes} | "
                f"Score: {ep_score:.2f} | Best: {best_score:.2f} | Epsilon: {epsilon:.3f}"
            )

    env.close()
    agent.save(str(model_path / "final.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN agent to descend straight down smoothly to the pad."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-buffer-size", type=int, default=1_000)
    parser.add_argument("--target-update-freq", type=int, default=1_000)
    parser.add_argument("--model-dir", type=str, default="checkpoints/land")
    args = parser.parse_args()

    train_land(
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
