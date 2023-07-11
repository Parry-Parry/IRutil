from abc import ABC
import logging
import json
import os

import torch
from ts.torch_handler.base_handler import BaseHandler

from terrierutil.llm.model.build import init_causallm

logger = logging.getLogger(__name__)

class TransformersLLMHandler(BaseHandler, ABC):
    def __init__(self):
        super(TransformersLLMHandler, self).__init__()
        self.initialized = False
        self.cfg = None

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        size = os.getenv('MODEL_SIZE')
        properties = ctx.system_properties

        model_dir = os.getenv(properties.model_yaml_config[size]['dir_env_var'])
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = init_causallm(model_dir=model_dir, tokenizer_dir=model_dir, device=self.device)

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True
    
    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        cfg = None
        for data in requests:
            input_text = data.get("data")
            _cfg = data.get("config")
            if _cfg is not None and cfg is None: 
                self.cfg = json.loads(_cfg)
                max_length = self.cfg["max_input_length"]

            if self.cfg is None: 
                self.cfg = {}
                max_length = 128

            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
        
            logger.info("Received text: '%s'", input_text)
            # preprocessing text for sequence_classification, token_classification or text_generation

            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=int(max_length),
                pad_to_max_length=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            input_ids_batch = None
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
           
        outputs = self.model.generate(
            input_ids_batch, **self.cfg['generation_params']
        )

        for i, x in enumerate(outputs):
            inferences.append(
                self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            )

        logger.info("Generated text: '%s'", inferences)

        return inferences


    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output
    
    def construct_input_ref(text, tokenizer, device):
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        # construct input token ids
        logger.info("text_ids %s", text_ids)
        logger.info("[tokenizer.cls_token_id] %s", [tokenizer.cls_token_id])
        input_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
        logger.info("input_ids %s", input_ids)

        input_ids = torch.tensor([input_ids], device=device)
        # construct reference token ids
        ref_input_ids = (
            [tokenizer.cls_token_id]
            + [tokenizer.pad_token_id] * len(text_ids)
            + [tokenizer.sep_token_id]
        )
        ref_input_ids = torch.tensor([ref_input_ids], device=device)
        # construct attention mask
        attention_mask = torch.ones_like(input_ids)
        return input_ids, ref_input_ids, attention_mask


_service = TransformersLLMHandler()

def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data, cfg = _service.preprocess(data)
        data = _service.inference(data, cfg)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e