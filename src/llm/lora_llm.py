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
        pass 

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer
    

