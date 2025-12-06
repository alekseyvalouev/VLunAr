#!/usr/bin/env python

import argparse
from pathlib import Path
import random

import gymnasium as gym
import h5py
import numpy as np

from models.dqn_class import DQNAgent
from environments.custom_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
)


class ExtraFlagsWrapper(gym.Wrapper):
    """Wrapper that adds extra visual flags during rendering without affecting observations."""
    
    def __init__(self, env, add_extra_flags=False, seed=None):
        super().__init__(env)
        self.add_extra_flags = add_extra_flags
        self.extra_flag_positions = []  # List of (x1, x2, y, color) tuples
        self.helipad_color = None
        self.helipad_color_name = None
        self.extra_flag_color_name = None
        self.rng = np.random.RandomState(seed)
        
        # Color options (RGB tuples)
        self.color_options = {
            'red': (255, 0, 0),
            'orange': (255, 165, 0),
            'yellow': (255, 255, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'purple': (128, 0, 128),
        }
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.add_extra_flags:
            # Access the unwrapped environment to get terrain info
            unwrapped = self.env.unwrapped
            
            # Get helipad location
            helipad_x1 = unwrapped.helipad_x1
            helipad_x2 = unwrapped.helipad_x2
            helipad_y = unwrapped.helipad_y
            
            # Get terrain information
            W = VIEWPORT_W / SCALE
            H = VIEWPORT_H / SCALE
            CHUNKS = 11
            chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
            
            # Find helipad chunk indices
            helipad_center = CHUNKS // 2
            helipad_start_idx = helipad_center - 2
            helipad_end_idx = helipad_center + 2
            
            # Available chunks for extra flags (not overlapping with helipad)
            # Left side: chunks 0 to helipad_start_idx-1 (exclusive of helipad)
            # Right side: chunks helipad_end_idx+1 to CHUNKS-1 (exclusive of helipad)
            left_chunks = list(range(0, helipad_start_idx))
            right_chunks = list(range(helipad_end_idx + 1, CHUNKS))
            
            # Randomly choose left or right side
            if len(left_chunks) > 0 and len(right_chunks) > 0:
                side = self.rng.choice(['left', 'right'])
                available_chunks = left_chunks if side == 'left' else right_chunks
            elif len(left_chunks) > 0:
                available_chunks = left_chunks
            elif len(right_chunks) > 0:
                available_chunks = right_chunks
            else:
                available_chunks = []
            
            if len(available_chunks) >= 1:
                # Pick a random chunk for the extra flags
                center_idx = self.rng.choice(available_chunks)
                
                # Place flags around this chunk (similar to helipad)
                flag_idx1 = max(0, center_idx - 1)
                flag_idx2 = min(CHUNKS - 1, center_idx + 1)
                
                extra_flag_x1 = chunk_x[flag_idx1]
                extra_flag_x2 = chunk_x[flag_idx2]
                
                # Estimate the terrain height at this location
                # Use helipad_y as a reasonable default (flat terrain)
                extra_flag_y = helipad_y
                
                # Randomly select colors for both flag pairs
                color_names = list(self.color_options.keys())
                self.rng.shuffle(color_names)
                helipad_color_name = color_names[0]
                extra_flag_color_name = color_names[1]
                
                self.helipad_color = self.color_options[helipad_color_name]
                extra_flag_color = self.color_options[extra_flag_color_name]
                
                # Store for rendering
                self.extra_flag_positions = [(extra_flag_x1, extra_flag_x2, extra_flag_y, extra_flag_color)]
                
                # Store color names for printing
                self.helipad_color_name = helipad_color_name
                self.extra_flag_color_name = extra_flag_color_name
                
                # Modify the unwrapped environment to use custom colors
                unwrapped.custom_helipad_color = self.helipad_color
                unwrapped.custom_extra_flags = self.extra_flag_positions
        
        return obs, info


def make_env(render_mode: str = "rgb_array", add_extra_flags: bool = False, seed: int = None):
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
    
    # Wrap with extra flags if requested
    if add_extra_flags:
        env = ExtraFlagsWrapper(env, add_extra_flags=True, seed=seed)
    
    return env


def run_land_agent(
    checkpoint_path: str,
    video_folder: str = "videos/land",
    episodes: int = 1,
    max_steps: int = 1000,
    add_extra_flags: bool = False,
    actions_folder: str = "actions/land",
):
    # base env (custom, starting above the pad)
    # Use None for seed to get random behavior each time
    env = make_env(render_mode="rgb_array", add_extra_flags=add_extra_flags, seed=None)

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
        name_prefix="land_agent",
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
        print(f"[LAND] Episode {ep+1}/{episodes} | Return: {ep_return:.1f} | Frames: {len(episode_frames)}")
    
    # Print flag colors if extra flags were added
    if add_extra_flags and isinstance(env.env, ExtraFlagsWrapper):
        wrapper = env.env
        if (hasattr(wrapper, 'helipad_color_name') and 
            hasattr(wrapper, 'extra_flag_color_name') and
            wrapper.helipad_color_name is not None and 
            wrapper.extra_flag_color_name is not None):
            print(f"\nFlag Colors:")
            print(f"  Helipad flags: {wrapper.helipad_color_name.capitalize()}")
            print(f"  Extra flags: {wrapper.extra_flag_color_name.capitalize()}")
    
    env.close()
    print(f"Videos saved to: {video_path.absolute()}")
    
    # Save frames to HDF5 format
    h5_file = video_path / "land_video.h5"
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
    parser = argparse.ArgumentParser(description="Test LAND agent (descending to the pad)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/strafe_down/best.pt",
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
    parser.add_argument(
        "--add-flags",
        action="store_true",
        default=False,
        help="Add extra pair of flags at random location with random colors",
    )
    args = parser.parse_args()

    run_land_agent(
        checkpoint_path=args.checkpoint,
        video_folder=args.video_folder,
        episodes=args.episodes,
        max_steps=args.max_steps,
        add_extra_flags=args.add_flags,
        actions_folder="actions/land",
    )


if __name__ == "__main__":
    main()
