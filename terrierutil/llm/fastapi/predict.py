#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
import torch
from terrierutil.llm.fastapi.config import CONFIG
import numpy as np


def preprocess(package: dict, text : List[str]) -> list:
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding
    :param package: dict from fastapi state including model and preocessing objects
    :param package: list of input to be proprocessed
    :return: list of proprocessed input
    """

    # preprocessing text for sequence_classification, token_classification or text_generation

    inputs = package["tokenizer"].encode_plus(
        text,
        max_length=CONFIG['max_input_length'],
        pad_to_max_length=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].cuda()

    return input_ids


def predict(package: dict, text : str, generation_params : dict) -> Tuple[str, np.ndarray]:
    """
    Run model and get result
    :param package: dict from fastapi state including model and preocessing objects
    :param package: list of input values
    :return: numpy array of model output
    """

    # process data
    X = preprocess([text])

    # run model
    model = package['model']
    with torch.no_grad():    
        outputs = model.generate(
            X, **generation_params
        ).cpu()

    texts = package["tokenizer"].decode(outputs, skip_special_tokens=True)[0]
    logits = outputs.logits.numpy()[0]

    return texts, logits