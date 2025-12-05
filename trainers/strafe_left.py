# %%

import argparse
from pathlib import Path
from typing import Optional
import math

import numpy as np

import gymnasium as gym  # still needed for types / wrappers if you want later

from models.dqn_class import DQNAgent
from environments.custom_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    FPS,
)

TASK_NAME = "strafe_left"

# --- Target horizontal speed (world units: "m/s") ---
# We want the lander to move left at about 1 m/s in Box2D units.
# The environment encodes x_dot in state as:
#   state[2] = vel.x * (VIEWPORT_W / SCALE / 2) / FPS
# So we compute the corresponding target value in state space.
VX_SCALE = (VIEWPORT_W / SCALE / 2.0) / FPS
TARGET_WORLD_VX = -1.0  # 1 m/s *to the left*
TARGET_STATE_VX = TARGET_WORLD_VX * VX_SCALE


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
    Create *custom* LunarLander environment (your version).

    We bypass gym.make("LunarLander-v3") and instantiate your class directly.
    We leave init_x/init_y as None here so we can randomize them per-episode.
    """
    env = LunarLander(
        render_mode=render_mode,
        continuous=False,           # DQN with discrete actions
        random_initial_force=False  # no random push; we want controlled strafing
    )
    return env


def shaped_reward_strafe_left(
    next_state: np.ndarray,
    target_y: float,
    start_x: float,
    # altitude / vertical motion
    w_height: float = 8.0,             # stronger height penalty
    w_vert_speed: float = 3.0,         # stronger vertical speed penalty
    # horizontal strafing
    w_horiz_speed: float = 3.0,
    w_progress: float = 0.1,
    w_right: float = 10.0,             # penalty for moving right of start_x
    # orientation
    w_angle: float = 4.0,              # keep angle near 0 (upright)
    w_ang_speed: float = 0.5,          # discourage spinning
    upside_down_penalty: float = 10.0, # big penalty if |angle| > pi/2
    step_penalty: float = 0.01,
) -> float:
    """
    Reward for strafing LEFT at ~1 m/s while:
      - maintaining altitude (near target_y),
      - staying upright (angle ~ 0),
      - and NEVER going right of the starting x position.

    next_state: [x, y, x_dot, y_dot, theta, theta_dot, leg1, leg2]
    target_y:   initial y at episode start (desired altitude)
    start_x:    initial x at episode start (do NOT go to the right of this)
    """

    x = next_state[0]
    y = next_state[1]
    x_dot = next_state[2]
    y_dot = next_state[3]
    angle = next_state[4]       # 0 is upright, +/- is tilt
    ang_vel = next_state[5]

    # --- Altitude control: stay close to initial height and avoid vertical motion ---
    height_error = y - target_y      # want this ~ 0
    vert_speed = y_dot               # want this ~ 0

    # --- Horizontal speed control: want x_dot ≈ TARGET_STATE_VX ---
    speed_error = x_dot - TARGET_STATE_VX

    # --- Progress term: being farther left (negative x) is slightly good globally ---
    progress_term = -x  # x normalized; more negative = better

    # --- Orientation: stay upright and avoid spinning ---
    angle_error = angle           # want 0 rad
    ang_vel_error = ang_vel       # want 0 rad/s

    # --- Relative horizontal position: penalize going right of start_x ---
    # If x > start_x, that's moving to the right; penalize (x - start_x)^2.
    delta_x_from_start = max(0.0, x - start_x)

    reward = (
        # vertical / altitude
        -w_height * (height_error ** 2)
        -w_vert_speed * (vert_speed ** 2)
        # horizontal strafing speed
        -w_horiz_speed * (speed_error ** 2)
        # global left progress
        + w_progress * progress_term
        # orientation
        -w_angle * (angle_error ** 2)
        -w_ang_speed * (ang_vel_error ** 2)
        # do NOT move right of starting x
        -w_right * (delta_x_from_start ** 2)
        # small per-step penalty to encourage efficiency
        - step_penalty
    )

    # Extra harsh penalty when upside-down (more than 90 degrees tilt)
    if abs(angle) > (math.pi / 2):
        reward -= upside_down_penalty

    return float(reward)


def train_strafe_left(
    episodes: int = 1000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 100_000,
    min_buffer_size: int = 1_000,
    target_update_freq: int = 1_000,
    model_dir: str = "checkpoints/strafe_left",
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

    # World dimensions in Box2D units
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE

    for ep in range(1, episodes + 1):
        # --- Randomize starting position in world coordinates ---
        # X: keep away from extreme edges a bit
        init_x = np.random.uniform(world_w * 0.25, world_w * 0.75)
        # Y: somewhere between mid-height and near the top, safely above terrain
        init_y = np.random.uniform(world_h * 0.5, world_h * 0.9)

        env.unwrapped.custom_init_x = init_x
        env.unwrapped.custom_init_y = init_y
        env.unwrapped.random_initial_force = False  # ensure no random kick

        state, _ = env.reset()
        done = False

        # Target altitude = initial y (whatever the random start is)
        target_y = state[1]
        # Record starting x in *state space* (normalized x), so we can penalize going right of it
        start_x = state[0]

        ep_score = 0.0
        epsilon = linear_epsilon_decay(ep, episodes)

        for t in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Use OUR custom reward only (env_reward is ignored)
            reward = shaped_reward_strafe_left(next_state, target_y, start_x)

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
        description=(
            "Train DQN agent to strafe LEFT at ~1 m/s while "
            "holding altitude, staying upright, and never moving right of its start."
        )
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-buffer-size", type=int, default=1_000)
    parser.add_argument("--target-update-freq", type=int, default=1_000)
    parser.add_argument("--model-dir", type=str, default="checkpoints/strafe_left")
    args = parser.parse_args()

    train_strafe_left(
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
