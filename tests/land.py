#!/usr/bin/env python

import argparse
from pathlib import Path

import gymnasium as gym

from models.dqn_class import DQNAgent
from environments.custom_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
)


def make_env(render_mode: str = "rgb_array"):
    """
    Create the custom LunarLander environment and start the lander
    above the helipad with no initial impulse, for the landing task.
    """
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE

    # Same helipad definition as in custom_lander.reset: helipad_y = H / 4
    helipad_y = world_h / 4.0

    # Start somewhere above the helipad.
    # You can tweak this height if you want a higher or lower approach.
    init_x = world_w / 2.0
    init_y = helipad_y + 4.0  # ~4 world units above the pad

    env = LunarLander(
        render_mode=render_mode,
        continuous=False,
        init_x=init_x,
        init_y=init_y,
        random_initial_force=False,  # no random kick; we want a clean descent
    )
    return env


def run_land_agent(
    checkpoint_path: str,
    video_folder: str = "videos/land",
    episodes: int = 1,
    max_steps: int = 1000,
):
    # base env (custom, starting above the pad)
    env = make_env(render_mode="rgb_array")

    # create video folder
    video_path = Path(video_folder)
    video_path.mkdir(parents=True, exist_ok=True)

    # wrap for video recording
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_path),
        episode_trigger=lambda ep: True,
        name_prefix="land_agent",
    )

    # agent setup
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(state, epsilon=0.0)  # greedy
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            ep_return += reward
            state = next_state
            steps += 1

        print(f"[LAND] Episode {ep+1}/{episodes} | Return: {ep_return:.1f}")

    env.close()
    print(f"Videos saved to: {video_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Test LAND agent (descending to the pad)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/land/best.pt",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="videos/land",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    run_land_agent(
        checkpoint_path=args.checkpoint,
        video_folder=args.video_folder,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
