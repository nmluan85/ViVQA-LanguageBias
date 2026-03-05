from omegaconf import OmegaConf
from underthesea import word_tokenize
from transformers import AutoTokenizer

from typing import List, Optional, Union
import transformers
from transformers.utils import TensorType
from lavis.processors.base_processor import BaseProcessor
from lavis.models.blip2_model.blip2_qformer import Blip2Qformer
from lavis.processors.blip_processor import (
    BlipCaptionProcessor, 
    BlipQuestionProcessor,
    BlipImageTrainProcessor,
    BlipImageEvalProcessor, 
    Blip2ImageTrainProcessor,
)
import torch

class Processor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = self.load_preprocess()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word", use_fast=False)

    def load_preprocess(self):
        config = OmegaConf.load(Blip2Qformer.default_config_path(model_type="pretrain"))
        preprocess_cfg = config.preprocess
        def _build_proc_from_cfg(cfg):
            if cfg is None:
                return BaseProcessor()
            if cfg.name == "blip_caption":
                return BlipCaptionProcessor.from_config(cfg)
            elif cfg.name == "blip_question":
                return BlipQuestionProcessor.from_config(cfg)
            elif cfg.name == "blip_image_train":
                return BlipImageTrainProcessor.from_config(cfg)
            elif cfg.name == "blip_image_eval":
                return BlipImageEvalProcessor.from_config(cfg)
            elif cfg.name == "blip2_image_train":
                return Blip2ImageTrainProcessor.from_config(cfg)
        
        vis_proc_cfg = preprocess_cfg.get("vis_processor")
        vis_eval_cfg = vis_proc_cfg.get("eval")
        vis_processors = _build_proc_from_cfg(vis_eval_cfg)
        return vis_processors

    def __call__(
        self,
        image,
        text: Union[str, List[str], List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False,
        truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ):
        image = self.image_processor(image.convert("RGB"))

        text = word_tokenize(text.lower(), format='text')
        text = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose
        )
        text['padding_mask'] = 1 - text['attention_mask']

        return {
            'image': image.to(self.device),
            'question': text['input_ids'].to(self.device),
            'padding_mask': text['padding_mask'].to(self.device)
        }