from abc import ABC
import logging
import ast

import torch
from torchserve.torch_handler.base_handler import BaseHandler

from terrierutil.llm.model.lora_llm import build_llama
from terrierutil.llm.util import LlamaConfig

logger = logging.getLogger(__name__)


class TransformersLlamaHandler(BaseHandler, ABC):
    def __init__(self):
        super(TransformersLlamaHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        cfg = LlamaConfig.build_llama_config(properties.get("model_config"), properties.get("adapter"), self.device)
        self.model, self.tokenizer = build_llama(cfg)

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
                cfg = _cfg

            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
        
            max_length = self.setup_config["max_length"]
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
     
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch), cfg

    def inference(self, input_batch, cfg):
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
        # Handling inference for sequence_classification.
        if self.setup_config["mode"] == "sequence_classification":
            predictions = self.model(input_ids_batch, attention_mask_batch)
            print(
                "This the output size from the Seq classification model",
                predictions[0].size(),
            )
            print("This the output from the Seq classification model", predictions)

            num_rows, _ = predictions[0].shape
            for i in range(num_rows):
                out = predictions[0][i].unsqueeze(0)
                y_hat = out.argmax(1).item()
                predicted_idx = str(y_hat)
                inferences.append(self.mapping[predicted_idx])
        
           
        outputs = self.model.generate(
            input_ids_batch, **cfg
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


_service = TransformersLlamaHandler()

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