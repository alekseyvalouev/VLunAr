import torch
import argparse
from PIL import Image
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainers.vla_trainer import train_vla

def generate_sample_data(num_samples=4):
    data = []
    for i in range(num_samples):
        # Random image (noise)
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Sample text
        prompt = f"User: Do task {i}\nModel:"
        text = f" I am doing task {i}"
        
        # Random state vector (dim 8 as per VLA class)
        state_vec = torch.randn(8)
        
        # Random actions (dim 4 as per ActProjector, length 10)
        # Assuming actions are token IDs for now, or just indices if ActProjector handles it.
        # ActProjector._build_act_tokens uses one_hot with num_classes=4.
        # So actions should be indices in [0, 3].
        # Let's assume we have a sequence of actions.
        # VLA._build_inputs_embeds expects 'actions' tensor.
        # ActProjector takes 'tokens'.
        # In VLA._build_inputs_embeds: act_embeds = self.act_projector(actions)
        # In ActProjector.forward: x = self._build_act_tokens(tokens) -> one_hot
        # So 'actions' passed to VLA should be indices [batch, seq_len] or [seq_len] if unbatched?
        # VLADataset returns item. collate_fn stacks them.
        # So item['actions'] should be [seq_len].
        actions = torch.randint(0, 4, (10,))
        
        data.append({
            'image': image,
            'prompt': prompt,
            'text': text,
            'state_vec': state_vec,
            'actions': actions
        })
    return data

def main():
    parser = argparse.ArgumentParser(description="Run VLA training with sample data")
    parser.add_argument("--ckpt", type=str, default="google/paligemma-3b-pt-224", help="Path or ID of the checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/vla_sample", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()

    print(f"Generating sample data...")
    data = generate_sample_data()
    
    print(f"Starting training with checkpoint: {args.ckpt}")
    # Note: This will try to download the model if not cached.
    # Ensure you have access/internet if using a gated model or remote ID.
    
    try:
        train_vla(
            ckpt_path=args.ckpt,
            data=data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        print("Sample training finished successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
