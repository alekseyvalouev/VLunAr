#!/usr/bin/env python

import argparse
from pathlib import Path
import zipfile

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

                # Use helipad_y as the terrain height (flat there)
                extra_flag_y = helipad_y

                # Randomly select colors for both flag pairs
                color_names = list(self.color_options.keys())
                self.rng.shuffle(color_names)
                helipad_color_name = color_names[0]
                extra_flag_color_name = color_names[1]

                self.helipad_color = self.color_options[helipad_color_name]
                extra_flag_color = self.color_options[extra_flag_color_name]

                # Store for rendering
                self.extra_flag_positions = [
                    (extra_flag_x1, extra_flag_x2, extra_flag_y, extra_flag_color)
                ]

                # Store color names for printing
                self.helipad_color_name = helipad_color_name
                self.extra_flag_color_name = extra_flag_color_name

                # Modify the unwrapped environment to use custom colors
                unwrapped.custom_helipad_color = self.helipad_color
                unwrapped.custom_extra_flags = self.extra_flag_positions

        return obs, info


def make_env(render_mode: str = "rgb_array", add_extra_flags: bool = False, seed: int = None):
    """
    Create the custom LunarLander environment.

    The lander:
    - Can spawn at any horizontal position (x in [0, world_w]),
    - But always spawns high enough above terrain so it is not inside a mountain
      or on the ground, and has a chance to fly and find the flags.
    """
    world_w = VIEWPORT_W / SCALE
    world_h = VIEWPORT_H / SCALE

    # Terrain heights are always in [0, H/2], so spawning above ~0.6 H is guaranteed safe.
    safe_min_y = world_h * 0.6
    safe_max_y = world_h * 0.9

    init_x = np.random.uniform(0.0, world_w)             # anywhere left → right
    init_y = np.random.uniform(safe_min_y, safe_max_y)   # high above any mountains

    env = LunarLander(
        render_mode=render_mode,
        continuous=False,
        init_x=init_x,
        init_y=init_y,
        random_initial_force=False,  # no random kick; agent starts stable
    )

    # Wrap with extra flags if requested
    if add_extra_flags:
        env = ExtraFlagsWrapper(env, add_extra_flags=True, seed=seed)

    return env


def generate_land_dataset(
    checkpoint_path: str,
    output_root: str = "land_dataset",
    episodes: int = 1000,
    max_steps: int = 1000,
    add_extra_flags: bool = False,
):
    """
    Run the LAND agent for `episodes` episodes, record frames + actions + 8D states,
    store everything in a single HDF5 file, then zip the whole dataset folder.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # We'll still record regular videos (mp4s) inside this root if you want them.
    video_path = output_root / "videos"
    video_path.mkdir(parents=True, exist_ok=True)

    # HDF5 dataset file
    h5_file_path = output_root / "land_dataset.h5"

    # Base env (custom spawn, safe above terrain)
    env = make_env(render_mode="rgb_array", add_extra_flags=add_extra_flags, seed=None)

    # Wrap for video recording (optional mp4s, but cheap to keep)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_path),
        episode_trigger=lambda ep: True,
        name_prefix="land_agent",
    )

    # Agent setup
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    # Open HDF5 file for frames + actions + states
    with h5py.File(h5_file_path, "w") as h5f:
        for ep in range(episodes):
            state, info = env.reset()
            done = False
            ep_return = 0.0
            steps = 0

            episode_actions = []
            episode_frames = []
            episode_states = []  # <-- store 8D state vectors

            # Initial state and frame
            episode_states.append(state)  # shape (8,)
            frame = env.render()
            episode_frames.append(frame)

            while not done and steps < max_steps:
                action = agent.select_action(state, epsilon=0.0)  # greedy
                episode_actions.append(int(action))

                next_state, reward, terminated, truncated, info = env.step(action)

                # Record next state and frame
                episode_states.append(next_state)
                frame = env.render()
                episode_frames.append(frame)

                done = terminated or truncated
                ep_return += reward
                state = next_state
                steps += 1

            # Convert to arrays
            frames_array = np.array(episode_frames, dtype=np.uint8)
            actions_array = np.array(episode_actions, dtype=np.int64)
            states_array = np.array(episode_states, dtype=np.float32)  # (T+1, 8)

            # Create group for this episode
            ep_group = h5f.create_group(f"episode_{ep + 1}")
            ep_group.create_dataset(
                "frames",
                data=frames_array,
                compression="gzip",
                compression_opts=4,
            )
            ep_group.create_dataset(
                "actions",
                data=actions_array,
                compression="gzip",
                compression_opts=4,
            )
            ep_group.create_dataset(
                "states",
                data=states_array,
                compression="gzip",
                compression_opts=4,
            )

            # Store metadata as attributes
            ep_group.attrs["num_frames"] = frames_array.shape[0]
            ep_group.attrs["num_actions"] = actions_array.shape[0]
            ep_group.attrs["num_states"] = states_array.shape[0]
            ep_group.attrs["frames_shape"] = frames_array.shape
            ep_group.attrs["states_shape"] = states_array.shape

            print(
                f"[LAND] Episode {ep + 1}/{episodes} | "
                f"Return: {ep_return:.1f} | "
                f"Frames: {frames_array.shape[0]} | "
                f"Actions: {actions_array.shape[0]} | "
                f"States: {states_array.shape[0]}"
            )

        # (Optionally) store some global metadata
        h5f.attrs["episodes"] = episodes
        h5f.attrs["max_steps"] = max_steps
        h5f.attrs["note"] = "LAND dataset: frames + actions + 8D states per episode"

    # Print flag colors if extra flags were added
    if add_extra_flags and isinstance(env.env, ExtraFlagsWrapper):
        wrapper = env.env
        if (
            hasattr(wrapper, "helipad_color_name")
            and hasattr(wrapper, "extra_flag_color_name")
            and wrapper.helipad_color_name is not None
            and wrapper.extra_flag_color_name is not None
        ):
            print("\nFlag Colors:")
            print(f"  Helipad flags: {wrapper.helipad_color_name.capitalize()}")
            print(f"  Extra flags: {wrapper.extra_flag_color_name.capitalize()}")

    env.close()
    print(f"\nHDF5 dataset saved to: {h5_file_path.absolute()}")
    print(f"Videos saved to: {video_path.absolute()}")

    # ------------------------------------------------------------
    # ZIP the whole dataset folder
    # ------------------------------------------------------------
    zip_path = output_root.with_suffix(".zip")
    print(f"\nZipping dataset folder into: {zip_path.absolute()}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_root.rglob("*"):
            # path.relative_to(output_root) ensures the zip has proper folder structure
            zf.write(path, arcname=path.relative_to(output_root))

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LAND dataset (frames + actions + 8D states in HDF5, zipped)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/land/land/best.pt",
        help="Path to trained LAND checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="land_dataset",
        help="Root folder for dataset (will contain HDF5 + videos and will be zipped).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of LAND episodes to generate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--add-flags",
        action="store_true",
        default=False,
        help="Add extra pair of flags at random location with random colors.",
    )
    args = parser.parse_args()

    generate_land_dataset(
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        episodes=args.episodes,
        max_steps=args.max_steps,
        add_extra_flags=args.add_flags,
    )


if __name__ == "__main__":
    main()
