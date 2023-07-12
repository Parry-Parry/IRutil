import json
import os

def send_request(address : str, text : str, generation_params : dict = {}, op : str = 'POST'):
    data_str = json.dumps({'text' : text, 'generation_params' : generation_params})
    args = f"curl -X '{op}' '{address}' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{data_str}'"
    return json.loads(os.popen(args).read())
    