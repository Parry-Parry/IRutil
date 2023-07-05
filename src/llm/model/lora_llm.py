from typing import Optional
from ..util import LlamaConfig
import os
import sys
from os.path import join
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

'''
Basic PEFT setup from https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)

def build_llama(config : LlamaConfig):
    llama_dir = join(os.getenv('LLAMA_DIR'), f'llama{config.size}')
    tokenizer = LlamaTokenizer.from_pretrained(llama_dir)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference
    model = LlamaForCausalLM.from_pretrained(llama_dir, **config.modelconfig)
    if config.adapter: 
        model = PeftModel.from_pretrained(model,
                                          config.adapter,
                                          torch_dtype=torch.float16,
                                        )
        
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer

def init_llama(config, lora_config : Optional[LoraConfig] = None, resume_from_checkpoint : Optional[str] = None):
    llama_dir = join(os.getenv('LLAMA_DIR'), f'llama{config.size}')
    tokenizer = LlamaTokenizer.from_pretrained(llama_dir)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference
    model = LlamaForCausalLM.from_pretrained(llama_dir, **config.modelconfig)

    if lora_config:
        model = get_peft_model(model, lora_config)
        model = prepare_model_for_int8_training(model)

        model.config.use_cache = False

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer

def tokenize(prompt, tokenizer, add_eos_token=True, cutoff_len=1024):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

def generate_and_tokenize_prompt(prompt, 
                                 tokenizer,
                                 train_on_inputs, 
                                 add_eos_token,
                                 cutoff_len,
                                 data_point):
        full_prompt = prompt.construct(**data_point)
        tokenized_full_prompt = tokenize(full_prompt, tokenizer=tokenizer, add_eos_token=add_eos_token, cutoff_len=cutoff_len)
        if not train_on_inputs:
            data_point.pop("output")
            user_prompt = prompt.construct(**data_point)
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  
        return tokenized_full_prompt

