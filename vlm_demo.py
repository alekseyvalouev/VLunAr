import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("Initializing LunarLander environment...")
    # Initialize the environment with rgb_array render mode to capture frames
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Reset the environment
    observation, info = env.reset(seed=42)
    
    print("Running simulation for a few steps...")
    # Run a few steps to get the lander in the air
    for _ in range(60):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    # Capture the current frame
    frame = env.render()
    env.close()
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(frame)
    
    # Save image for inspection
    image.save("lander_frame.png")
    print("Frame saved as 'lander_frame.png'")

    print("Loading VLM (google/paligemma-3b-pt-224)...")
    model_id = "google/paligemma-3b-pt-224"
    
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare prompt
    prompt = "describe the image"
    
    print(f"Analyzing image with prompt: '{prompt}'...")
    
    # Process inputs
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_len = model_inputs["input_ids"].shape[-1]

    # Generate response
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        
    print("-" * 20)
    print("Model Response:")
    print(decoded)
    print("-" * 20)

if __name__ == "__main__":
    main()
