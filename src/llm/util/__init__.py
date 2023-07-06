from dataclasses import dataclass, asdict
from json import dumps
from typing import List
import os

@dataclass
class LlamaConfig:
    size : str = '7B'
    modelconfig : dict = {}
    device : str = 'cuda'

    @staticmethod
    def build_llama_config(modelconfig : dict, 
                           device : str = 'cuda'):
        size = modelconfig.pop('size', '7B')
        return LlamaConfig(size=size, modelconfig=modelconfig, device=device)

@dataclass
class Log:
    epoch : int
    loss : dict
    val_metrics : dict

    @property
    def __dict__(self):
        return asdict(self)

    @property
    def json(self):
        return dumps(self.__dict__)

@dataclass
class LogStore:
    logs : List[Log]
    test_metrics : dict

    @property
    def __dict__(self):
        return asdict(self)

    @property
    def json(self):
        return dumps(self.__dict__)
    
def init_out(dir : str, subdirs : List[str] = ['models']):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    for subdir in subdirs:
        if not os.path.exists(os.path.join(dir, subdir)):
            os.makedirs(os.path.join(dir, subdir), exist_ok=True)
    
def dump_logs(logs : LogStore, dir : str):
    import json
    from os.path import join
    with open(join(dir, 'test.json'), 'w') as f:
        json.dump(logs.test_metrics, f, indent=4)
    with open(join(dir, 'logs.json'), 'w') as f:
        json.dump(logs.json, f, default=lambda o : o.__dict__, indent=4)