"""
train_recurrent_core.py
Autonomous training of the 2-layer recurrent core (Layers 13 & 14) on RTX 3050.
Uses 4-bit QLoRA on layers 13 and 14 with unrolled recurrent forward pass (T=2).
Target: 500 steps on FineWeb-Edu dataset.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    print("=== Starting Autonomous Recurrent Core Training (RTX 3050, 6GB VRAM) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    
    # 1. Load Tokenizer & Model Config
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    print(f"Loading config and tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg_model = AutoConfig.from_pretrained(model_id)

    # 2. Configure 4-bit Quantization and CPU Offload for Embeddings/Head to strictly respect 6GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    device_map = {
        "model.embed_tokens": "cpu",
        "model.norm": 0,
        "lm_head": "cpu",
    }
    for i in range(cfg_model.num_hidden_layers):
        device_map[f"model.layers.{i}"] = 0

    print("Loading 4-bit quantized base model (layers on GPU, embeddings/lm_head in CPU RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    
    # 3. Configure LoRA targeted strictly to the recurrent core layers (13 & 14)
    target_modules = [
        "model.layers.13.self_attn.q_proj", "model.layers.13.self_attn.k_proj",
        "model.layers.13.self_attn.v_proj", "model.layers.13.self_attn.o_proj",
        "model.layers.13.mlp.gate_proj", "model.layers.13.mlp.up_proj", "model.layers.13.mlp.down_proj",
        "model.layers.14.self_attn.q_proj", "model.layers.14.self_attn.k_proj",
        "model.layers.14.self_attn.v_proj", "model.layers.14.self_attn.o_proj",
        "model.layers.14.mlp.gate_proj", "model.layers.14.mlp.up_proj", "model.layers.14.mlp.down_proj",
    ]
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    from accelerate.hooks import remove_hook_from_module
    from transformers.masking_utils import create_causal_mask

    raw_base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    remove_hook_from_module(raw_base.model.embed_tokens, recurse=True)
    remove_hook_from_module(raw_base.lm_head, recurse=True)

    # 4. Load Training Data
    data_file = "/home/cune/data/fineweb_edu_sample.jsonl"
    print(f"Loading training data from {data_file}...")
    texts = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item.get("text", ""))
                if len(texts) >= 1000:
                    break
    print(f"Loaded {len(texts)} training documents.")

    # 5. Training Loop Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    max_steps = 500
    batch_size = 1
    grad_accum_steps = 4
    seq_len = 256
    
    model.train()
    optimizer.zero_grad()
    
    step = 0
    doc_idx = 0
    total_loss = 0.0
    
    print(f"Starting 500 optimization steps (seq_len={seq_len}, grad_accum={grad_accum_steps})...")
    
    output_dir = "/home/cune/llama.cpp/models/recurrent_core_lora"
    os.makedirs(output_dir, exist_ok=True)
    
    loss_fn = nn.CrossEntropyLoss()

    raw_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    embed_tokens = raw_model.model.embed_tokens
    layers = raw_model.model.layers
    norm = raw_model.model.norm
    lm_head = raw_model.lm_head
    hidden_size = cfg_model.hidden_size

    while step < max_steps:
        text = texts[doc_idx % len(texts)]
        doc_idx += 1
        
        enc = tokenizer(text, max_length=seq_len, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"] # on CPU
        attention_mask = enc.get("attention_mask", None)
            
        # 1. Embedding on CPU -> GPU 0
        inputs_embeds = embed_tokens(input_ids)
        hidden_states = inputs_embeds.to(0)
        curr_seq_len = input_ids.shape[1]
        position_ids = torch.arange(curr_seq_len, device=0).unsqueeze(0)
        position_embeddings = raw_model.model.rotary_emb(hidden_states, position_ids)

        attn_mask_dev = attention_mask.to(0) if attention_mask is not None else None
        causal_mask = create_causal_mask(
            config=cfg_model,
            inputs_embeds=hidden_states,
            attention_mask=attn_mask_dev,
            past_key_values=None,
            position_ids=position_ids
        )

        # 2. Prelude: layers 0..12
        for i in range(13):
            out = layers[i](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False
            )
            hidden_states = out[0] if isinstance(out, tuple) else out

        # 3. Anchor & Recurrent loop T=2 over layers 13..14
        anchor_e = hidden_states
        h = hidden_states
        recurrent_a = 0.90
        recurrent_b = 0.10
        recurrent_gate = 1.00

        for t in range(2):
            combined = torch.nn.functional.rms_norm(h + anchor_e, (hidden_size,), eps=1e-6)
            cur = combined
            for i in [13, 14]:
                out = layers[i](
                    cur,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False
                )
                cur = out[0] if isinstance(out, tuple) else out
            delta_thought = cur - combined
            h = (recurrent_a * h + recurrent_b * anchor_e) + recurrent_gate * delta_thought

        hidden_states = h

        # 4. Coda: layers 15..27
        for i in range(15, 28):
            out = layers[i](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False
            )
            hidden_states = out[0] if isinstance(out, tuple) else out

        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states.to(lm_head.weight.device))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous().to(logits.device)
        sample_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss = sample_loss / grad_accum_steps
        loss.backward()
        total_loss += sample_loss.item()
        
        if (doc_idx % grad_accum_steps) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            
            if step % 5 == 0:
                avg_loss = total_loss / 5.0
                vram_used = torch.cuda.memory_allocated() / (1024**2)
                print(f"Step {step:3d}/{max_steps} | Loss: {avg_loss:.4f} | VRAM: {vram_used:.1f} MB")
                total_loss = 0.0
                
            if step % 100 == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}")
                model.save_pretrained(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    final_path = os.path.join(output_dir, "recurrent_core_final")
    model.save_pretrained(final_path)
    print(f"=== Optimization Complete! Final adapter saved to {final_path} ===")

if __name__ == "__main__":
    main()
