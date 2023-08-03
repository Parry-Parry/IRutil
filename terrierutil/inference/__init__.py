import json
import requests
from typing import List
import numpy as np
from dataclasses import dataclass

@dataclass
class APIOutput:
    def __init__(self, package) -> None:
        self.text = package['results']['text']
        self.logits = np.array(package['results']['logits'])

def send_request(url : List[str], text : str, generation_params : dict = {}, op : str = 'POST'):
    header = {"Content-type": "application/json",
              "accept": 'application/json'} 
    payload = json.dumps({'text' : text, 'generation_params' : generation_params})
    response_decoded_json = requests.post(url, data=payload, headers=header, verify=False)
    result = response_decoded_json.json()
    if result['error']:
        return result
    else:
        return APIOutput(result)