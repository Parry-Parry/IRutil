import logging 
from typing import List
import json

class Prompt:
    def __init__(self, prompt : str, params : List[str] = None, name='Standard Prompt', description='Standard Prompt'):
        self.prompt = prompt
        self.params = params
        self.name = name
        self.description = description

        if params:
            for param in params: assert f'{{{param}}}' in prompt, f'Param {param} not found in prompt {prompt}'
        
    def __str__(self):
        return self.prompt

    def __repr__(self):
        return f'Prompt(prompt={self.prompt}, params={self.params})'
    
    @staticmethod
    def fromjson(json_str):
        return json.loads(json_str, object_hook=lambda x: Prompt(**x))
    
    @staticmethod
    def fromstring(string : str, params=None, name='Standard Prompt', description='Standard Prompt'):
        return Prompt(prompt=string, params=params, name=name, description=description)
    
    def tojson(self):
        return json.dumps(self, default=lambda x: x.__dict__, 
            sort_keys=True, indent=4)
    
    def construct(self, kwargs):
        for key in kwargs.keys(): 
            if key not in self.params:
                logging.warning(f'Key {key} not found in params {self.params}')
                kwargs.pop(key)
        return self.prompt.format(**kwargs)
    
    def batch_construct(self, params : List[dict], num_proc : int):
        '''
        Ensure that params is a list of dicts and large enough to justify overhead of multiprocessing
        '''
        if num_proc is None:
            return [self.construct(**param) for param in params]
        from multiprocessing import Pool, cpu_count
        if num_proc is None: num_proc = cpu_count()
        with Pool(num_proc) as p:
            return p.map(self.construct, params)
    
    def __call__(self, inp, num_proc=None):
        if isinstance(inp, list):
            return self.batch_construct(inp, num_proc=num_proc)
        else:
            return self.construct(**inp)