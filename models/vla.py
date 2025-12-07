import torch
from torch import nn
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

# Normally, when we merge image tokens, we get
# <image> ... <image> (x 256 for 224x224 image) <prompt> ... <prompt> <text> ... <text>
# Usually this is given and then we predict                           ^^^^^^^^^^^^^^^^^
# Instead, we will train on 
# <image> ... <image> <state> <prompt> ... <prompt> <text> ... <text> <act> <act> .... <act>
# The following is predicted w/ next token pred.    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# our training data thus needs to look like that. We will also need to add <state> embeddings and <act> embeddings.

class VLA(nn.Module):
    def __init__(self, ckpt : str):
        super().__init__()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cuda:1"
        )
        self.processor = AutoProcessor.from_pretrained(ckpt)

        self.d_model = self.model.config.hidden_size
        self.state_projector = StateProjector(self.d_model).to(self.model.device)
        self.act_projector = ActProjector(self.d_model).to(self.model.device)
    
    @torch.no_grad()
    def _build_inputs_embeds(self, image: Image, prompt: str | list[str], text: str | list[str], state_vec: torch.Tensor, actions: torch.Tensor):
        # 1) Processor gives you pixel_values + tokenized text
        # We need to process prompt and text together to get the full sequence, 
        # but we also need to know where the prompt ends to mask it.
        
        # Handle batching for text inputs
        if isinstance(prompt, list) and isinstance(text, list):
            full_text = [p + t for p, t in zip(prompt, text)]
        else:
            full_text = prompt + text 
        
        # Process images and full text
        inputs = self.processor(text=full_text, images=image, return_tensors="pt", padding=True).to(self.model.device)
        
        # We also need to tokenize just the prompt to know its length in tokens
        # We use add_special_tokens=False for the prompt length check to avoid double counting if possible,
        # but the processor usually adds them. Let's rely on the processor's consistency.
        # A safer way is to tokenize both and see. 
        # For simplicity in this VLA context, let's assume standard behavior:
        prompt_inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.model.device)
        n_prompt_tokens = prompt_inputs.input_ids.shape[1]
        
        # 2) Vision -> projected image tokens (soft tokens)
        vision_out = self.model.vision_tower(inputs["pixel_values"])
        image_embeds = self.model.multi_modal_projector(vision_out.last_hidden_state)
        image_embeds = image_embeds * (self.model.config.hidden_size ** -0.5) 
        image_embeds = image_embeds.to(self.model.device)

        # 3) State -> soft tokens
        state_embeds = self.state_projector(state_vec.to(self.model.device)).to(image_embeds.dtype)

        # 4) Action -> soft tokens
        act_embeds = self.act_projector(actions.to(self.model.device)).to(image_embeds.dtype)

        # 5) Text token embeddings
        input_ids = inputs["input_ids"]  # [B, T]
        text_embeds = self.model.get_input_embeddings()(input_ids).to(self.model.device)
        print(image_embeds.shape, state_embeds.shape, text_embeds.shape, act_embeds.shape)

        # 6) Concat: [img | state | text | act]
        # Note: 'text' here in embeddings includes both prompt and response
        inputs_embeds = torch.cat([image_embeds, state_embeds, text_embeds, act_embeds], dim=1)

        # 7) Labels
        # We want to predict: <text> (response part) and <act>
        # We mask: <image>, <state>, <prompt>
        
        B = input_ids.shape[0]
        n_img = image_embeds.shape[1]
        n_state = state_embeds.shape[1]
        n_act = act_embeds.shape[1]
        n_text_total = input_ids.shape[1]
        
        # Create mask tensor filled with -100
        # Shape: [B, n_img + n_state + n_prompt]
        # Wait, n_prompt_tokens might include special tokens that are also in n_text_total.
        # If prompt is "Hello", input_ids might be [BOS, Hello]. 
        # If full text is "Hello World", input_ids might be [BOS, Hello, World].
        # n_prompt_tokens would be 2. n_text_total would be 3. 
        # We want to mask indices 0 and 1. So we mask `n_prompt_tokens` amount.
        # This is a heuristic and depends on the tokenizer, but is standard for simple concatenation.
        
        # Labels for the prefix (image + state + prompt) -> Masked
        prefix_mask = torch.full((B, n_img + n_state + n_prompt_tokens), -100, device=self.model.device, dtype=torch.long)
        
        # Labels for the response (text part) -> input_ids[n_prompt_tokens:]
        # We need to slice input_ids to get the response part
        resp_labels = input_ids[:, n_prompt_tokens:]
        
        # Labels for actions -> actions
        # We assume 'actions' passed in are the token IDs for actions
        act_labels = actions.to(self.model.device, dtype=torch.long)
        
        labels = torch.cat([prefix_mask, resp_labels, act_labels], dim=1)

        # 8) Attention Mask
        # inputs["attention_mask"] covers the text part. 
        # We need to add 1s for image, state, and actions.
        text_attn = inputs["attention_mask"].to(self.model.device) # [B, n_text_total]
        img_attn = torch.ones((B, n_img), device=self.model.device, dtype=torch.long)
        state_attn = torch.ones((B, n_state), device=self.model.device, dtype=torch.long)
        act_attn = torch.ones((B, n_act), device=self.model.device, dtype=torch.long)
        
        attn = torch.cat([img_attn, state_attn, text_attn, act_attn], dim=1)

        sizes = {
            "n_img": n_img,
            "n_state": n_state,
            "n_prompt": n_prompt_tokens,
            "n_text": n_text_total,
            "n_act": n_act
        }

        return inputs_embeds, attn, labels, sizes

    @torch.no_grad()
    def forward(self, image, prompt, text, state_vec, actions):
        inputs_embeds, attention_mask, labels, _ = self._build_inputs_embeds(image, prompt, text, state_vec, actions)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def predict(self, image, prompt, text, state_vec, actions):
        inputs_embeds, attention_mask, labels, sizes = self._build_inputs_embeds(image, prompt, text, state_vec, actions)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        
        # Decode output
        logits = out.logits
        
        # Indices
        n_prefix = sizes["n_img"] + sizes["n_state"] + sizes["n_prompt"]
        n_text = sizes["n_text"] - sizes["n_prompt"] # Response length
        n_act = sizes["n_act"]
        
        # 1) Text
        # Logits for text start after prefix
        text_logits = logits[:, n_prefix : n_prefix + n_text, :]
        text_ids = torch.argmax(text_logits, dim=-1)
        decoded_text = self.processor.batch_decode(text_ids, skip_special_tokens=False)
        
        # 2) Actions
        # Logits for actions are at the end
        act_logits = logits[:, -n_act:, :]
        pred_actions = torch.argmax(act_logits, dim=-1)
        
        return decoded_text, pred_actions
        
class StateProjector(nn.Module):
    """
    Project state into VLM token space. 
    """
    # Matmul: [batch_size, d_state] @ [d_state, d_model] -> [batch_size, d_model]
    # Unsqueeze -> [batch_size, 1, d_model]
    def __init__(self, d_model : int):
        super().__init__()
        self.d_state = 8
        self.norm = nn.LayerNorm(self.d_state)
        self.proj = nn.Linear(self.d_state, d_model)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        return x.unsqueeze(1)

class ActProjector(nn.Module):
    """
    Project action into VLM token space. 
    """
    # Matmul: 
    # [batch_size, n_act_tokens, d_act] @ [d_act, d_model] -> [batch_size, n_act_tokens, d_model]
    def __init__(self, d_model : int):
        super().__init__()
        self.d_tokens = 4
        self.norm = nn.LayerNorm(self.d_tokens)
        self.proj = nn.Linear(self.d_tokens, d_model)
    
    def _build_act_tokens(self, tokens):
        # Tokens are like [batch_size, n_act_tokens], we should convert to one-hot dim 4
        return torch.nn.functional.one_hot(tokens, num_classes=self.d_tokens).to(dtype=self.proj.weight.dtype)
    
    def forward(self, tokens):
        x = self._build_act_tokens(tokens)
        x = self.norm(x)
        x = self.proj(x)
        return x

if __name__ == "__main__":
    model_id = "google/paligemma-3b-pt-224"
    vla = VLA(model_id)


    b_size = 2
    horizon = 10

    image = [Image.open("lander_frame.png")] * b_size
    prompt = ["Fly up and to the right "] * b_size
    text = ["fly up", "fly right"]
    state_vec = torch.zeros((b_size, 8))
    actions = torch.zeros((b_size, horizon)).to(dtype=torch.long)

    out = vla(image, prompt, text, state_vec, actions)

    print(f"Loss: {out.loss}")
    
    decoded_text, pred_actions = vla.predict(image, prompt, text, state_vec, actions)
    print("Decoded Text:", decoded_text)
    print("Predicted Actions:", pred_actions)
