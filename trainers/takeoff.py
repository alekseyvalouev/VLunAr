# %%

import argparse
from pathlib import Path
from typing import Optional
import math

import numpy as np

import gymnasium as gym  # still useful for wrappers / types

from models.dqn_class import DQNAgent
from environments.custom_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    FPS,
    LEG_DOWN,  # to place the lander on the surface
)

TASK_NAME = "takeoff"

# --- Target vertical speed (world units: "m/s") ---
# We want the lander to go UP at about 1 m/s in Box2D units.
# The environment encodes y_dot in state as:
#   state[3] = vel.y * (VIEWPORT_H / SCALE / 2) / FPS
# So we compute the corresponding target value in state space.
VY_SCALE = (VIEWPORT_H / SCALE / 2.0) / FPS
TARGET_WORLD_VY = +1.0  # 1 m/s upwards
TARGET_STATE_VY = TARGET_WORLD_VY * VY_SCALE


def linear_epsilon_decay(
    episode: int,
    max_episodes: int,
    start_eps: float = 1.0,
    end_eps: float = 0.05,
) -> float:
    frac = min(1.0, episode / max_episodes)
    return start_eps + frac * (end_eps - start_eps)


def make_env(render_mode: Optional[str] = None):
    """
    Create *custom* LunarLander environment.

    We instantiate your LunarLander directly and control init_x/init_y and
    random_initial_force from the trainer so we can start *on the surface*.
    """
    env = LunarLander(
        render_mode=render_mode,
        continuous=False,
        random_initial_force=False,  # no random kick; we want a clean liftoff
    )
    return env


def shaped_reward_takeoff(
    next_state: np.ndarray,
    target_x: float,
    w_vy: float = 5.0,              # match vertical velocity to target
    w_horiz_pos: float = 3.0,       # stay above target_x
    w_horiz_speed: float = 2.0,     # minimize sideways drift
    w_tilt: float = 4.0,            # stay upright
    w_tilt_speed: float = 0.5,      # don't spin
    upside_down_penalty: float = 10.0,
    step_penalty: float = 0.01,
) -> float:
    """
    Reward for clean vertical takeoff at ~1 m/s upward.

    next_state: [x, y, x_dot, y_dot, theta, theta_dot, leg1, leg2]
    target_x:   initial x at episode start -> vertical line reference
    """

    x = next_state[0]
    x_dot = next_state[2]
    y_dot = next_state[3]
    theta = next_state[4]
    theta_dot = next_state[5]

    # --- Vertical speed: want y_dot ≈ TARGET_STATE_VY ---
    vy_error = y_dot - TARGET_STATE_VY

    # --- Stay roughly above initial x ---
    horiz_pos_error = x - target_x  # want this ~ 0
    # squared error to penalize drifting either side
    horiz_pos_term = horiz_pos_error ** 2

    # --- Minimize sideways drift (horizontal speed) ---
    horiz_speed = x_dot  # want ~ 0
    horiz_speed_term = horiz_speed ** 2

    # --- Orientation: stay upright and don't spin ---
    tilt = theta          # want 0 rad
    tilt_speed = theta_dot  # want 0 rad/s

    reward = (
        -w_vy * (vy_error ** 2)
        -w_horiz_pos * (horiz_pos_term)
        -w_horiz_speed * (horiz_speed_term)
        -w_tilt * (tilt ** 2)
        -w_tilt_speed * (tilt_speed ** 2)
        -step_penalty
    )

    # Extra harsh penalty when upside-down (more than 90 degrees tilt)
    if abs(theta) > (math.pi / 2):
        reward -= upside_down_penalty

    return float(reward)


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

    best_score = -np.inf

    # Compute world dimensions and helipad height
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE
    helipad_y = world_h / 4.0  # same formula as in custom_lander.reset

    # Place body center so legs rest on the helipad:
    # ground at helipad_y, legs extend LEG_DOWN/SCALE below lander body,
    # body polygon bottom is roughly 10 / SCALE below center.
    # So center_y ≈ ground + (LEG_DOWN + 10) / SCALE
    body_bottom_offset = 10.0  # from LANDER_POLY lowest vertex (-10)
    base_y = helipad_y + (LEG_DOWN + body_bottom_offset) / SCALE
    base_x = world_w / 2.0

    for ep in range(1, episodes + 1):
        # For now, always take off from the middle of the helipad area
        env.unwrapped.custom_init_x = base_x
        env.unwrapped.custom_init_y = base_y
        env.unwrapped.random_initial_force = False  # make sure there's no impulse

        state, _ = env.reset()
        done = False

        # Reference vertical line = starting x (normalized)
        target_x = state[0]

        ep_score = 0.0
        epsilon = linear_epsilon_decay(ep, episodes)

        for t in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Ignore env_reward; use our shaped reward for clean vertical takeoff
            reward = shaped_reward_takeoff(next_state, target_x)

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
                f"Score (shaped): {ep_score:.2f} | Best: {best_score:.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()
    agent.save(str(model_path / "final.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN agent for clean vertical takeoff from the surface at ~1 m/s."
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
