import argparse
from pathlib import Path
from typing import Optional, Tuple
import math

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

TASK_NAME = "vertical_down"

# --- Target vertical speed (world units: "m/s") ---
# We want the lander to move straight DOWN more slowly, e.g. about 0.3 m/s in Box2D units.
# The environment encodes y_dot in state as:
#   state[3] = vel.y * (VIEWPORT_H / SCALE / 2) / FPS
VY_SCALE = (VIEWPORT_H / SCALE / 2.0) / FPS
TARGET_WORLD_VY = -0.1  # 0.3 m/s downward (slower descent)
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
    env = LunarLander(
        render_mode=render_mode,
        continuous=False,
        random_initial_force=False,
    )
    return env


def shaped_reward_vertical_down(
    next_state: np.ndarray,
    start_x: float,
    descent_started: bool,
    done: bool,
    *,
    # per-step penalties
    step_penalty: float = 1.0,
    w_x_offset: float = 10.0,      # lateral position penalty
    w_angle: float = 5.0,          # basic tilt penalty
    w_x_vel: float = 20.0,         # penalize horizontal velocity (left/right motion)
    w_vy_track: float = 2.0,       # track vertical speed toward target downward speed each step
    # EXTRA: strongly penalize any orientation that isn't basically straight up
    upright_tol_angle: float = 0.02,   # ~1.1 degrees
    w_upright_excess: float = 100.0,   # strong penalty once outside the tiny upright band
    # going back up after descent starts
    upward_penalty: float = 200.0,
    # terminal rewards/penalties
    landing_good_reward: float = 2000.0,
    landing_bad_penalty: float = 2000.0,
    landing_tol_x: float = 0.03,
    landing_tol_angle: float = 0.2,   # radians (~11 degrees)
    w_landing_speed: float = 50.0,    # penalize deviation from target vertical speed on landing
) -> Tuple[float, bool]:
    """
    Reward spec:

    Goal: smooth, straight-down descent at a slow vertical speed (e.g. ~0.3 m/s),
    no horizontal motion, then a clean stop on the ground.

    - Every step:
        * step_penalty < 0 to discourage long episodes.
        * Penalize sideways position (x_offset^2).
        * Penalize tilt (angle^2) + extra penalty if not basically upright.
        * Penalize horizontal velocity (vx^2) to avoid left/right movement.
        * Penalize deviation of vertical speed from target (y_dot - TARGET_STATE_VY)^2
          to encourage a smooth, slow descent the whole way.
    - Once descent has started (y_dot < 0), going up (y_dot > 0) is heavily punished.
    - On episode end:
        * If landed (legs touching), penalize deviation from the target vertical speed.
        * If also near start_x and upright -> big landing_good_reward.
        * Otherwise -> landing_bad_penalty.
    """
    x = next_state[0]
    vx = next_state[2]
    y_dot = next_state[3]
    angle = next_state[4]
    leg1 = next_state[6]
    leg2 = next_state[7]

    # --- Base: time penalty ---
    reward = -step_penalty

    # --- Sideways position penalty (stay directly under start_x) ---
    x_offset = x - start_x
    reward -= w_x_offset * (x_offset ** 2)

    # --- Tilt penalty ---
    reward -= w_angle * (angle ** 2)

    # --- Strong penalty for any deviation from upright orientation ---
    abs_angle = abs(angle)
    if abs_angle > upright_tol_angle:
        reward -= w_upright_excess * ((abs_angle - upright_tol_angle) ** 2 + 1.0)

    # --- Penalize any horizontal motion (left/right) ---
    reward -= w_x_vel * (vx ** 2)

    # --- Track vertical speed toward target each step (smooth, slow descent) ---
    speed_error_step = y_dot - TARGET_STATE_VY
    reward -= w_vy_track * (speed_error_step ** 2)

    # --- Track if we've started descending ---
    if not descent_started and y_dot < -0.01:
        descent_started = True

    # After descent starts, ANY upward velocity is smashed
    if descent_started and y_dot > 0.0:
        reward -= upward_penalty * (y_dot ** 2 + 1.0)

    # --- Terminal reward: evaluate final landing quality ---
    if done:
        landed = (leg1 > 0.5 or leg2 > 0.5)

        if landed:
            lateral = abs(x_offset)
            speed_error = y_dot - TARGET_STATE_VY
            x_ok = lateral < landing_tol_x
            angle_ok = abs_angle < landing_tol_angle

            # Penalize landed speed that differs from the target slow downward speed
            reward -= w_landing_speed * (speed_error ** 2)

            if angle_ok and x_ok:
                # Good landing: big positive
                reward += landing_good_reward
            else:
                # Landed but wrong place or too tilted -> hard negative
                reward -= landing_bad_penalty
        else:
            # Episode ended but no leg contact (crash, timeout, etc.) -> hard negative
            reward -= landing_bad_penalty

    return float(reward), descent_started


def train_vertical_down(
    episodes: int = 1000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 100_000,
    min_buffer_size: int = 1_000,
    target_update_freq: int = 1_000,
    model_dir: str = "land/vertical_down",  # save under land folder
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

    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE

    for ep in range(1, episodes + 1):
        # Random starting position
        init_x = np.random.uniform(world_w * 0.25, world_w * 0.75)
        init_y = np.random.uniform(world_h * 0.5, world_h * 0.9)

        env.unwrapped.custom_init_x = init_x
        env.unwrapped.custom_init_y = init_y
        env.unwrapped.random_initial_force = False

        state, _ = env.reset()
        done = False

        start_x = state[0]
        descent_started = False

        ep_score = 0.0
        epsilon = linear_epsilon_decay(ep, episodes)

        for t in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward, descent_started = shaped_reward_vertical_down(
                next_state,
                start_x,
                descent_started,
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
                f"Score (shaped): {ep_score:.2f} | Best: {best_score:.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()
    agent.save(str(model_path / "final.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train DQN agent to go straight down and TOUCH DOWN with "
            "a slow vertical speed (e.g. ~0.3 m/s), upright, directly below start_x, "
            "with no left/right motion."
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
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".checkpoints/land",
    )
    args = parser.parse_args()

    train_vertical_down(
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
# %%