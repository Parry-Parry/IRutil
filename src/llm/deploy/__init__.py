def prepare_packet(text : str, config : dict):
    return dict(data=text, config=config.__dict__)