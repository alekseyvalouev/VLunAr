import argparse
from pathlib import Path
from typing import Optional

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

# --- Vertical speed scaling for ~1 m/s downward ---
TARGET_WORLD_VY = -1.0  # 1 m/s downward
VY_SCALE = (VIEWPORT_H / SCALE / 2.0) / FPS
TARGET_STATE_VY = TARGET_WORLD_VY * VY_SCALE


def linear_epsilon_decay(ep, max_ep, start=1.0, end=0.05):
    frac = min(1.0, ep / max_ep)
    return start + frac * (end - start)


def make_env(render_mode: Optional[str] = None):
    """Make custom environment with no initial impulse."""
    return LunarLander(
        render_mode=render_mode,
        continuous=False,
        random_initial_force=False,
    )


def shaped_reward_strafe_down(
    prev_state: np.ndarray,
    next_state: np.ndarray,
    done: bool,
    landing_bonus: float = 30.0,
    crash_penalty: float = 60.0,
    w_progress: float = 6.0,   # get closer to ground
    w_tilt: float = 10.0,      # punish tilt
    w_vspeed: float = 10.0,    # punish not being at -1 m/s
    w_hspeed: float = 6.0,     # punish horizontal speed (want mostly vertical)
    step_penalty: float = 0.01,
) -> float:
    """
    Reward function for "strafe down":

      - Reward moving closer to the ground (y -> 0 from above)
      - Penalize ANY tilt
      - Penalize horizontal speed (encourage mostly vertical motion)
      - Penalize deviation from ~1 m/s downward
      - Small terminal bonus for gentle touchdown, penalty for crash
    """

    # Positions
    y_prev = prev_state[1]
    y = next_state[1]

    # Clamp above-ground positions
    y_prev_c = max(0.0, y_prev)
    y_c = max(0.0, y)

    # 1) Reward progress toward the ground
    # > 0 if we moved closer to y = 0 (assuming y > 0 is above ground)
    progress_reward = w_progress * (y_prev_c - y_c)

    # 2) Penalize tilt (stay upright)
    theta = next_state[4]
    tilt_penalty = w_tilt * (theta ** 2)

    # 3) Penalize deviation from -1 m/s downward
    vy = next_state[3]
    vy_error = vy - TARGET_STATE_VY
    vspeed_penalty = w_vspeed * (vy_error ** 2)

    # 4) Penalize horizontal speed (we want mostly vertical descent)
    vx = next_state[2]
    hspeed_penalty = w_hspeed * (vx ** 2)

    reward = (
        progress_reward
        - tilt_penalty
        - vspeed_penalty
        - hspeed_penalty
        - step_penalty
    )

    # Legs contact info
    leg1 = next_state[6]
    leg2 = next_state[7]
    feet_on_ground = (leg1 > 0.5) or (leg2 > 0.5)

    if done:
        # If episode ended and we are near ground with low speed & tilt,
        # treat as a "nice controlled descent".
        near_ground = abs(y) < 0.15
        upright = abs(theta) < 0.2
        slow_v = abs(vy) < 0.2
        small_hspeed = abs(vx) < 0.2

        gentle_touch = feet_on_ground and near_ground and upright and slow_v and small_hspeed

        if gentle_touch:
            reward += landing_bonus
        else:
            reward -= crash_penalty

    return float(reward)


def train_strafe_down(
    episodes=1000,
    max_steps=1000,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_size=100_000,
    min_buffer_size=1000,
    target_update_freq=1000,
    model_dir="checkpoints/strafe_down",
):
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim,
        action_dim,
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

    # World dimensions for random spawn
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE
    helipad_y = world_h / 4.0

    for ep in range(1, episodes + 1):
        # Random start for robustness: anywhere horizontally, reasonably high up
        start_x_world = np.random.uniform(world_w * 0.1, world_w * 0.9)
        start_y_world = np.random.uniform(helipad_y + 3.0, helipad_y + 7.0)

        env.unwrapped.custom_init_x = start_x_world
        env.unwrapped.custom_init_y = start_y_world
        env.unwrapped.random_initial_force = False

        state, _ = env.reset()

        ep_reward = 0.0
        epsilon = linear_epsilon_decay(ep, episodes)

        for t in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, env_rew, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            shaped = shaped_reward_strafe_down(state, next_state, done)

            agent.push_transition(state, action, shaped, next_state, done)
            agent.optimize_model()

            state = next_state
            ep_reward += shaped

            if done:
                break

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(str(model_path / "best.pt"))

        if ep % 10 == 0:
            print(
                f"[{TASK_NAME}] Ep {ep}/{episodes} | "
                f"Return: {ep_reward:.1f} | Best: {best_reward:.1f} "
                f"| Eps: {epsilon:.3f}"
            )

    env.close()
    agent.save(str(model_path / "final.pt"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a DQN agent to perform a slow, controlled vertical descent "
            "from arbitrary starting positions."
        )
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-buffer-size", type=int, default=1000)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--model-dir", type=str, default="checkpoints/strafe_down")
    args = parser.parse_args()

    train_strafe_down(
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
