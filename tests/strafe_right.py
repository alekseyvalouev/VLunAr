#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import h5py

from models.dqn_class import DQNAgent
from environments.custom_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
)


def make_env(
    render_mode: str = "rgb_array",
    init_x_frac: float = 0.5,
    init_y_frac: float = 0.5,
    random_initial_force: bool = False,
):
    """
    Create the custom LunarLander environment and start the lander at a
    configurable position.

    init_x_frac, init_y_frac are fractions of the world width/height:
      0.0 -> left/bottom edge
      0.5 -> center
      1.0 -> right/top edge

    By default, this is (0.5, 0.5) -> middle-middle.
    """
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE

    init_x = world_w * init_x_frac
    init_y = world_h * init_y_frac

    env = LunarLander(
        render_mode=render_mode,
        continuous=False,
        init_x=init_x,
        init_y=init_y,
        random_initial_force=random_initial_force,
    )
    return env


def run_strafe_right_agent(
    checkpoint_path: str,
    video_folder: str = "videos/strafe_right",
    episodes: int = 1,
    max_steps: int = 1000,
    init_x_frac: float = 0.5,
    init_y_frac: float = 0.5,
    random_initial_force: bool = False,
    actions_folder: str = "actions/strafe_right",
):
    # base env (custom, with chosen spawn position)
    env = make_env(
        render_mode="rgb_array",
        init_x_frac=init_x_frac,
        init_y_frac=init_y_frac,
        random_initial_force=random_initial_force,
    )

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
        name_prefix="strafe_right_agent",
    )

    # agent setup
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    all_episode_actions = []  # Store actions from all episodes
    all_episode_frames = []  # Store frames from all episodes
    
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        episode_actions = []  # Store actions for this episode
        episode_frames = []  # Store frames for this episode
        
        # Render initial frame
        frame = env.render()
        episode_frames.append(frame)

        while not done and steps < max_steps:
            action = agent.select_action(state, epsilon=0.0)  # greedy
            episode_actions.append(int(action))  # Store the action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Render frame after action
            frame = env.render()
            episode_frames.append(frame)

            done = terminated or truncated
            ep_return += reward
            state = next_state
            steps += 1

        all_episode_actions.append(episode_actions)
        all_episode_frames.append(episode_frames)
        print(f"[STRAFE_RIGHT] Episode {ep+1}/{episodes} | Return: {ep_return:.1f} | Frames: {len(episode_frames)}")

    env.close()
    print(f"Videos saved to: {video_path.absolute()}")
    
    # Save frames to HDF5 format
    h5_file = video_path / "strafe_right_video.h5"
    with h5py.File(h5_file, 'w') as f:
        for ep_idx, frames in enumerate(all_episode_frames):
            # Convert list of frames to numpy array
            frames_array = np.array(frames, dtype=np.uint8)
            # Create dataset for this episode
            f.create_dataset(
                f"episode_{ep_idx + 1}/frames",
                data=frames_array,
                compression="gzip",
                compression_opts=4
            )
            # Store metadata
            f[f"episode_{ep_idx + 1}"].attrs['num_frames'] = len(frames)
            f[f"episode_{ep_idx + 1}"].attrs['num_actions'] = len(all_episode_actions[ep_idx])
            f[f"episode_{ep_idx + 1}"].attrs['shape'] = frames_array.shape
    
    print(f"HDF5 video saved to: {h5_file.absolute()}")
    
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
    parser = argparse.ArgumentParser(description="Test STRAFE RIGHT agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/strafe_right/best.pt",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="videos/strafe_right",
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
        "--init-x-frac",
        type=float,
        default=0.5,
        help="Initial X position as fraction of world width (0.0 left, 0.5 center, 1.0 right)",
    )
    parser.add_argument(
        "--init-y-frac",
        type=float,
        default=0.5,
        help="Initial Y position as fraction of world height (0.0 bottom, 0.5 middle, 1.0 top)",
    )
    parser.add_argument(
        "--random-initial-force",
        action="store_true",
        help="If set, apply the environment's random initial impulse to the lander",
    )
    args = parser.parse_args()

    run_strafe_right_agent(
        checkpoint_path=args.checkpoint,
        video_folder=args.video_folder,
        episodes=args.episodes,
        max_steps=args.max_steps,
        init_x_frac=args.init_x_frac,
        init_y_frac=args.init_y_frac,
        random_initial_force=args.random_initial_force,
        actions_folder="actions/strafe_right",
    )


if __name__ == "__main__":
    main()
