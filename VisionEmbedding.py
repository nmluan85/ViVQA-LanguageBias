import timm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from omegaconf import OmegaConf
from lavis.models.blip2_model.blip2_qformer import Blip2Qformer
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.blip_processor import (
    BlipCaptionProcessor, 
    BlipQuestionProcessor,
    BlipImageTrainProcessor,
    BlipImageEvalProcessor, 
    Blip2ImageTrainProcessor,
)

class Blip2ViTExtractor(nn.Module):
    def __init__(self):
        super(Blip2ViTExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Blip2Qformer.from_pretrained(model_type="pretrain").to(self.device)
        self.processor = self.load_preprocess()
        self.model_name = "blip2"
    
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
    
    def forward(self, *images):
        self.model.eval()
        images = torch.stack([self.processor(image.convert("RGB")).to(self.device) for image in images])
        image_features = self.model.extract_features(samples={"image": images}, mode="image").image_embeds
        return image_features
    
class EfficientnetExtractor(nn.Module):
    def __init__(self, model_name):
        super(EfficientnetExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EfficientNet.from_pretrained(model_name, advprop=True).to(self.device)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 768))
        self.model_name = model_name
        self.processor = self.load_preprocess()

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
    
    def forward(self, *images):
        self.model.eval()
        images_transformed = torch.stack([self.processor(image.convert('RGB')).to(self.device) for image in images])
        batch_size = images_transformed.size(0)
        x = self.model.extract_features(images_transformed)
        x = self.pooling1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.pooling2(x)
        x = x.reshape(batch_size, x.shape[1], -1)
        return x
        
class Blip2EfficientExtractor(nn.Module):
    def __init__(self):
        super(Blip2EfficientExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # BLIP-2
        self.model_blip2 = Blip2Qformer.from_pretrained(model_type="pretrain").to(self.device)
        if self.device == "cpu" or self.device == torch.device("cpu"):
            self.model_blip2 = self.model_blip2.float()
        self.model_blip2.eval()

        # EfficientNet
        self.model_efficientnet = EfficientNet.from_pretrained("efficientnet-b7", advprop=True).to(self.device)
        self.model_efficientnet.eval()
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, images):
        # Extract global
        global_features = self.model_blip2.extract_features(samples={"image": images}, mode="image").image_embeds # 32 x 768

        # Extract local
        local_features = self.model_efficientnet.extract_features(images) # 2560 x 7 x 7
        local_features = self.pooling1(local_features) # 2560 x 1 x 32
        local_features = local_features.permute(0, 3, 2, 1) # 32 x 1 x 2560
        local_features = self.pooling2(local_features) # 32 x 1 x 768
        batch_size = images.shape[0] 
        local_features = local_features.reshape(batch_size, local_features.shape[1], -1) #  32 x 768

        # Combine global & local features
        v = torch.cat([global_features, local_features], dim=1)
        return v
    

