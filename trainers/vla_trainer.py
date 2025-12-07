import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig, TrainingArguments, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import bitsandbytes as bnb
from models.vla import VLA
import os

class VLADataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Mock implementation - replace with actual data loading logic
        # Expected data format: dict with image, prompt, text, state_vec, actions
        item = self.data[idx]
        return item

def collate_fn(batch):
    # Custom collate function to handle batching of VLA inputs
    # This needs to be adapted based on the actual data structure
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    texts = [item['text'] for item in batch]
    state_vecs = torch.stack([item['state_vec'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    
    return {
        'image': images,
        'prompt': prompts,
        'text': texts,
        'state_vec': state_vecs,
        'actions': actions
    }

def train_vla(
    ckpt_path: str,
    data: list,
    output_dir: str = "checkpoints/vla_lora",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    # 1. Load Model with QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32
    )

    print(f"Loading VLA model from {ckpt_path}...")
    vla = VLA(ckpt_path)
    
    # Apply quantization to the underlying PaliGemma model
    # Note: VLA wraps PaliGemma. We need to apply QLoRA to the inner model usually, 
    # or if we load the inner model with quantization first.
    # The VLA class loads the model in __init__. 
    # To use QLoRA properly, we should ideally pass the quantization config to from_pretrained.
    # Since VLA hardcodes the loading, we might need to modify VLA or patch it.
    # For now, let's assume we can modify the VLA instance or reload the inner model.
    # Actually, VLA.model is the PaliGemmaForConditionalGeneration.
    # We can try to replace it or reload it if needed, but standard QLoRA flow 
    # usually loads the model with the config.
    
    # Let's re-load the inner model with quantization for this trainer script
    # This is a bit inefficient but safe without modifying VLA code heavily.
    
    print("Reloading inner model with 4-bit quantization...")
    vla.model = PaliGemmaForConditionalGeneration.from_pretrained(
        ckpt_path,
        quantization_config=bnb_config,
        device_map="auto", # Let accelerate handle it
        torch_dtype=torch.float32
    )
    
    # Ensure projectors are on the same device and dtype as the model
    # Note: prepare_model_for_kbit_training might cast model to something else, 
    # but usually it keeps it as is or casts to compute_dtype.
    vla.state_projector.to(vla.model.device, dtype=torch.float32)
    vla.act_projector.to(vla.model.device, dtype=torch.float32)
    
    # Prepare for k-bit training
    vla.model = prepare_model_for_kbit_training(vla.model, use_gradient_checkpointing=False)
    
    # 2. Configure LoRA
    # Target modules for PaliGemma/Siglip/Gemma usually include q_proj, k_proj, v_proj, o_proj
    # We also want to train the projectors: state_projector, act_projector.
    # They are standard linear layers, so we can either full fine-tune them or LoRA them.
    # Since they are small, full fine-tuning is often better, but LoRA is fine too.
    # Let's add LoRA to attention layers and maybe MLP.
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM" # PaliGemma is technically conditional generation but acts like LM
    )
    
    vla.model = get_peft_model(vla.model, peft_config)
    vla.model.print_trainable_parameters()
    
    # We also need to make sure our custom projectors are trainable.
    # prepare_model_for_kbit_training might freeze everything.
    # get_peft_model unfreezes LoRA adapters.
    # We need to manually unfreeze projectors.
    for param in vla.state_projector.parameters():
        param.requires_grad = True
    for param in vla.act_projector.parameters():
        param.requires_grad = True
        
    # Monkeypatch forward to ensure inputs_embeds requires grad
    # This is needed because passing inputs_embeds to a frozen model (even with LoRA)
    # sometimes fails to track gradients if not explicitly set, especially when bypassing embedding layer.
    import types
    def forward_with_grad(self, image, prompt, text, state_vec, actions):
        inputs_embeds, attention_mask, labels, _ = self._build_inputs_embeds(image, prompt, text, state_vec, actions)
        inputs_embeds.requires_grad_(True)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    
    vla.forward = types.MethodType(forward_with_grad, vla)

    # 3. Data
    dataset = VLADataset(data, vla.processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(vla.parameters(), lr=learning_rate)
    
    # 5. Training Loop
    vla.train()
    print("Starting training...")
    
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move batch to device (handled inside VLA forward mostly, but good to be explicit if needed)
            # VLA forward takes raw inputs and handles device movement for some, 
            # but expects tensors for state/action to be on device or moves them.
            
            # Forward pass
            outputs = vla(
                image=batch['image'],
                prompt=batch['prompt'],
                text=batch['text'],
                state_vec=batch['state_vec'],
                actions=batch['actions']
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        epoch_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        vla.model.save_pretrained(epoch_dir) # Saves LoRA adapters
        torch.save(vla.state_projector.state_dict(), os.path.join(epoch_dir, "state_projector.pt"))
        torch.save(vla.act_projector.state_dict(), os.path.join(epoch_dir, "act_projector.pt"))
        print(f"Saved checkpoint to {epoch_dir}")

    print("Training complete.")

def run_inference(
    ckpt_path: str,
    adapter_path: str,
    image,
    prompt: str,
    state_vec: torch.Tensor,
    action_len: int = 10 # Dummy arg for now, inference logic depends on loop
):
    # Load base model
    vla = VLA(ckpt_path)
    
    # Load LoRA adapters
    vla.model = PeftModel.from_pretrained(vla.model, adapter_path)
    
    # Load projectors
    # Assuming they are in the same dir as adapter
    vla.state_projector.load_state_dict(torch.load(os.path.join(adapter_path, "state_projector.pt")))
    vla.act_projector.load_state_dict(torch.load(os.path.join(adapter_path, "act_projector.pt")))
    
    vla.eval()
    
    # For inference, we usually need a loop to generate actions one by one or chunk by chunk
    # The VLA.predict method provided in the file seems to do a single forward pass and decode.
    # Let's use that for now.
    
    # Build inputs for generation
    # We need to manually build embeddings for [image, state, prompt]
    # This logic mimics parts of VLA._build_inputs_embeds but for generation prefix
    
    # 1. Process inputs
    inputs = vla.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(vla.model.device)
    
    # 2. Vision embeddings
    vision_out = vla.model.vision_tower(inputs["pixel_values"])
    image_embeds = vla.model.multi_modal_projector(vision_out.last_hidden_state)
    image_embeds = image_embeds * (vla.model.config.hidden_size ** -0.5)
    
    # 3. State embeddings
    state_embeds = vla.state_projector(state_vec.to(vla.model.device)).to(image_embeds.dtype)
    
    # 4. Text embeddings (prompt only)
    input_ids = inputs["input_ids"]
    text_embeds = vla.model.get_input_embeddings()(input_ids)
    
    # 5. Concatenate: [img | state | prompt]
    inputs_embeds = torch.cat([image_embeds, state_embeds, text_embeds], dim=1)
    
    # 6. Attention mask
    # We need to construct attention mask for the prefix
    B = input_ids.shape[0]
    n_img = image_embeds.shape[1]
    n_state = state_embeds.shape[1]
    n_text = input_ids.shape[1]
    
    img_attn = torch.ones((B, n_img), device=vla.model.device, dtype=torch.long)
    state_attn = torch.ones((B, n_state), device=vla.model.device, dtype=torch.long)
    text_attn = inputs["attention_mask"]
    
    attention_mask = torch.cat([img_attn, state_attn, text_attn], dim=1)
    
    # 7. Generate
    # We expect the model to generate text response then actions
    # We can set max_new_tokens to cover both
    # Note: This assumes the model learns to output text then actions
    
    outputs = vla.model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=128, # Adjust as needed
        use_cache=True
    )
    
    # Decode outputs
    # outputs from generate with inputs_embeds usually contains only new tokens if we don't pass input_ids
    # But PaliGemma might behave differently. Let's assume it returns generated IDs.
    
    # We need to separate text and action tokens.
    # This is tricky without knowing the exact delimiter or length.
    # For now, let's assume the last `action_len` tokens are actions if we enforce length,
    # or we decode everything and try to parse.
    # Given the VLA structure, it predicts <text> ... <text> <act> ... <act>
    # The actions are likely special tokens or just tokens at the end.
    # Let's return the raw decoded text and the last few tokens as actions.
    
    decoded_text = vla.processor.batch_decode(outputs, skip_special_tokens=True)
    
    # Assuming last action_len tokens are actions
    # We need to be careful if outputs includes input_ids (it shouldn't if we passed inputs_embeds only)
    pred_actions = outputs[:, -action_len:]
    
    return decoded_text, pred_actions

if __name__ == "__main__":
    # Example usage (commented out)
    # data = [] # Load your data here
    # train_vla("google/paligemma-3b-pt-224", data)
    pass
