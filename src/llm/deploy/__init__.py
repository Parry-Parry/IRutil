def prepare_packet(text : str, config : LlamaConfig):
    return dict(text=text, config=config.__dict__)