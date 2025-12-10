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
TARGET_WORLD_VX = -1.5  # 1 m/s *to the left*
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
        random_initial_force=True,
        init_x=np.random.uniform(VIEWPORT_W / SCALE / 1.7, VIEWPORT_W / SCALE / 1.1),
        init_y=np.random.uniform(VIEWPORT_H / SCALE / 2, VIEWPORT_H / SCALE / 1.1)
    )
    return env


def shaped_reward_strafe_left(
    next_state: np.ndarray,
    target_y: float,
    start_x: float,
    # STRONG penalties for unwanted behavior
    w_height_change: float = 50.0,      # VERY STRONG: any height change is bad
    w_vert_speed: float = 40.0,         # VERY STRONG: any vertical motion is bad
    w_move_right: float = 100.0,        # EXTREMELY STRONG: moving right is forbidden
    # Smooth leftward motion
    w_speed_match: float = 20.0,        # STRONG: match target leftward speed
    w_speed_smoothness: float = 15.0,   # penalize jerky speed changes
    # Orientation (less critical but still important)
    w_angle: float = 5.0,               # stay upright
    w_ang_speed: float = 2.0,           # don't spin
    upside_down_penalty: float = 20.0,  # big penalty if upside down
    step_penalty: float = 0.01,
) -> float:
    """
    Reward for smooth, constant-velocity leftward strafing while:
      - STRONGLY punishing ANY height change (up or down)
      - EXTREMELY punishing ANY rightward movement
      - Rewarding smooth, constant leftward velocity at target speed
      - Penalizing jerky or erratic motion

    next_state: [x, y, x_dot, y_dot, theta, theta_dot, leg1, leg2]
    target_y:   initial y at episode start (must maintain this altitude)
    start_x:    initial x at episode start (must NEVER exceed this)
    """

    x = next_state[0]
    y = next_state[1]
    x_dot = next_state[2]
    y_dot = next_state[3]
    angle = next_state[4]
    ang_vel = next_state[5]

    # --- STRONG PENALTY: Any height change from target ---
    height_error = abs(y - target_y)  # absolute deviation from target height
    height_penalty = w_height_change * (height_error ** 2)

    # --- STRONG PENALTY: Any vertical motion (up or down) ---
    vert_speed_penalty = w_vert_speed * (y_dot ** 2)

    # --- EXTREME PENALTY: Moving right of starting position ---
    # If x > start_x, we're moving right - this is FORBIDDEN
    rightward_drift = max(0.0, x - start_x)
    move_right_penalty = w_move_right * (rightward_drift ** 2)
    
    # Additional penalty for positive x velocity (moving right)
    if x_dot > 0:
        move_right_penalty += w_move_right * (x_dot ** 2)

    # --- Smooth leftward velocity matching ---
    # We want x_dot ≈ TARGET_STATE_VX (constant leftward speed)
    speed_error = x_dot - TARGET_STATE_VX
    speed_match_penalty = w_speed_match * (speed_error ** 2)
    
    # --- Penalize speed jerkiness ---
    # If speed deviates significantly from target, it's jerky
    # This is already captured by speed_match_penalty, but we can add
    # an extra term for very large deviations (jerky motion)
    if abs(speed_error) > 0.3:  # threshold for "jerky"
        jerkiness_penalty = w_speed_smoothness * (abs(speed_error) ** 2)
    else:
        jerkiness_penalty = 0.0

    # --- Orientation: stay upright and stable ---
    angle_penalty = w_angle * (angle ** 2)
    ang_vel_penalty = w_ang_speed * (ang_vel ** 2)

    # --- Combine all penalties ---
    reward = (
        -height_penalty           # STRONG: no height changes
        -vert_speed_penalty       # STRONG: no vertical motion
        -move_right_penalty       # EXTREME: never move right
        -speed_match_penalty      # match target leftward speed
        -jerkiness_penalty        # smooth motion, not jerky
        -angle_penalty            # stay upright
        -ang_vel_penalty          # don't spin
        -step_penalty             # small efficiency penalty
    )

    # Extra harsh penalty when upside-down
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
