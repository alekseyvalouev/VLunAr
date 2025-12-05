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
    LEG_DOWN,
)


def make_env(render_mode: str = "rgb_array", max_height: float = 0.7):
    """
    Create the custom LunarLander environment and start the lander
    resting on the surface (legs on the helipad), with no initial impulse.
    """
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE

    # Terrain / helipad in your env: helipad_y = H / 4
    helipad_y = world_h / 4.0

    # LANDER_POLY bottom is at -10, so body_bottom_offset ~= 10
    body_bottom_offset = 10.0

    # Center of the lander such that the legs rest on the helipad:
    # ground_y + (LEG_DOWN + body_bottom_offset) / SCALE
    init_x = world_w / 2.0
    init_y = helipad_y + (LEG_DOWN + body_bottom_offset) / SCALE

    env = LunarLander(
        render_mode=render_mode,
        continuous=False,
        init_x=init_x,
        init_y=init_y,
        random_initial_force=False,  # no random kick; we want a clean takeoff
        max_height=max_height,
    )
    return env


def run_takeoff_agent(
    checkpoint_path: str,
    video_folder: str = "videos/takeoff",
    episodes: int = 1,
    max_steps: int = 1000,
    max_height: float = 0.7,
    actions_folder: str = "actions/takeoff",
):
    # base env (custom, starting on the floor)
    env = make_env(render_mode="rgb_array", max_height=max_height)

    # create video folder
    video_path = Path(video_folder)
    video_path.mkdir(parents=True, exist_ok=True)
    
    # create actions folder
    actions_path = Path(actions_folder)
    actions_path.mkdir(parents=True, exist_ok=True)

    # wrap for video recording
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_path),
        episode_trigger=lambda ep: True,
        name_prefix="takeoff_agent",
    )

    # agent setup
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    all_episode_actions = []  # Store actions from all episodes
    
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        episode_actions = []  # Store actions for this episode

        while not done and steps < max_steps:
            action = agent.select_action(state, epsilon=0.0)  # greedy
            episode_actions.append(int(action))  # Store the action
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            ep_return += reward
            state = next_state
            steps += 1

        all_episode_actions.append(episode_actions)
        print(f"[TAKEOFF] Episode {ep+1}/{episodes} | Return: {ep_return:.1f}")

    env.close()
    print(f"Videos saved to: {video_path.absolute()}")
    
    # Save actions to file
    actions_file = actions_path / "actions.txt"
    with open(actions_file, 'w') as f:
        for ep_idx, actions in enumerate(all_episode_actions):
            f.write(f"Episode {ep_idx + 1}:\n")
            f.write(f"Actions: {actions}\n")
            f.write(f"Total actions: {len(actions)}\n")
            f.write("\n")
    
    print(f"Actions saved to: {actions_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Test TAKEOFF agent (starting on the floor)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/takeoff/best.pt",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="videos/takeoff",
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
    parser.add_argument(
        "--max-height",
        type=float,
        default=0.7,
    )
    args = parser.parse_args()

    run_takeoff_agent(
        checkpoint_path=args.checkpoint,
        video_folder=args.video_folder,
        episodes=args.episodes,
        max_steps=args.max_steps,
        max_height=args.max_height,
        actions_folder="actions/takeoff",
    )


if __name__ == "__main__":
    main()
