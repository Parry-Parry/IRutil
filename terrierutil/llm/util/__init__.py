from dataclasses import dataclass, asdict
from json import dumps
from typing import List
import os
import json
from io import BytesIO 
from urllib.parse import urlencode

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

'''
def send_request_pycurl(address : str, text : str, generation_config : dict = {}, **kwargs):
    crl = pycurl.Curl() 
    buffer = BytesIO()
    crl.setopt(crl.WRITEDATA, buffer)
    crl.setopt(crl.URL, address)
    data_str = json.dumps({'data' : text, 'config' : {'generation_params' : generation_config, **kwargs}})
    pf = urlencode(data_str)
    crl.setopt(crl.POSTFIELDS, pf)
    crl.perform()
    crl.close()

    body = buffer.getvalue()
    return body.decode('utf-8')
'''

def send_request(address : str, text : str, generation_config : dict = {}, **kwargs):
    max_input = kwargs.pop('max_input', 128)
    data_str = json.dumps({'data' : text, 'config' : {'max_input_length': max_input, 'generation_params' : generation_config, **kwargs}})
    pf = urlencode(data_str)
    return os.popen(f'curl -X POST -d {pf} {address}').read()