#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Text2TextGenerationPipeline,
)

HFTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


class CustomText2TextGenerationPipeline(Text2TextGenerationPipeline):
    """Adds an `input_device` attribute to the pipeline and copies input tensors to it.
    This is necessary to use the pipeline with a model initialised directly on the GPU.
    """

    def __init__(self, model: DictConfig, tokenizer: DictConfig, *args, **kwargs):
        self.input_device = int(kwargs.get("input_device", -1))
        try:
            kwargs.pop("input_device")
        except KeyError:
            pass
        if int(self.input_device) == -1:  # cpu loading
            model: PreTrainedModel = instantiate(model)
            tokenizer: HFTokenizer = instantiate(tokenizer)
        else:
            with torch.device(f"{self.input_device}"):
                model: PreTrainedModel = instantiate(model)
                tokenizer: HFTokenizer = instantiate(tokenizer)
        super().__init__(model=model, tokenizer=tokenizer, *args, **kwargs)

    def _forward(self, model_inputs, **generate_kwargs):
        in_b, input_length = model_inputs["input_ids"].shape
        generate_kwargs["min_length"] = generate_kwargs.get(
            "min_length", self.model.config.min_length
        )
        generate_kwargs["max_length"] = generate_kwargs.get(
            "max_length", self.model.config.max_length
        )
        # copy the input tensors to device. we need to do this
        # manually since self.device has to be `cpu`
        # so that we can pass the model that is already on the GPU to the pipeline
        if self.input_device != -1:
            model_inputs = self._ensure_tensor_on_device(
                model_inputs, device=self.input_device
            )
        self.check_inputs(
            input_length, generate_kwargs["min_length"], generate_kwargs["max_length"]
        )
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        out_b = output_ids.shape[0]
        output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])

        return {"output_ids": output_ids}
