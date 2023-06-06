from dataclasses import dataclass

@dataclass
class LlamaConfig:
    size : str = '7B'
    modelconfig : dict = {}
    generationconfig : dict = {}
    adapter : str = None
    device : str = 'cuda'

    @staticmethod
    def build_llama_config(modelconfig : dict,
                           generationconfig : dict,
                           adapter : str = None,  
                           device : str = 'cuda'):
        size = modelconfig.pop('size', '7B')
        return LlamaConfig(size=size, modelconfig=modelconfig, generationconfig=generationconfig, adapter=adapter, device=device)

def build_lora_config():
    pass