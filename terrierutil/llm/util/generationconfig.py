import json

CONTRAST_CREATIVE = {
    'top_k' : 10,
    'do_sample' : True,
    'penalty_alpha' : 0.5,
    'temperature' : 2.0,
    'top_p' : 0.75,
}
    
BEAM_SEARCH = {
    'num_beams' : 10,
    'num_return_sequences' : 1,
    'no_repeat_ngram_size' : 1,
    'remove_invalid_values' : True,
    'top_p' : 0.75,
}
    
GREEDY_SMOOTH = {
    'num_beams': 1,
    'do_sample': False,
    'temperature': 2.0,
    'top_p' : 0.75,
}

CONTRAST_DETERMINISTIC = {
    'top_k' : 10,
    'do_sample' : False,
    'penalty_alpha' : 0.5,
    'temperature' : 0.1,
    'top_p' : 0.75,
}

GREEDY_DETERMINISTIC = {
    'num_beams': 1,
    'do_sample': False,
    'temperature': 0.1,
    'top_p' : 0.75,
}

def load_generation_config(type, **kwargs):
    types = {
    "contrast_creative": CONTRAST_CREATIVE,
    "beam_search":BEAM_SEARCH,
    "greedy_smooth": GREEDY_SMOOTH,
    "contrast_deterministic": CONTRAST_DETERMINISTIC,
    "greedy_deterministic": GREEDY_DETERMINISTIC,
    }
    return json.dumps({**types[type], **kwargs})