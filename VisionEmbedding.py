import timm
import torch
import torch.nn as nn
import numpy as np
import cv2
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
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers, 
    FastRCNNOutputs, 
    fast_rcnn_inference_single_image
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

class BottomUpFasterRCNNExtractor(nn.Module):
    def __init__(
        self,
        weights_path,
        k=36,
        score_thresh=0.2,
        nms_thresh=0.6,
        config_file="/kaggle/working/py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml",
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = k
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.config_file = config_file
        self.weights_path = weights_path

        self.predictor, self.model = self._build_model()
        self.model.eval()

    def _build_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_thresh
        cfg.MODEL.DEVICE = self.device

        predictor = DefaultPredictor(cfg)
        model = predictor.model
        return predictor, model

    def _pil_to_bgr_numpy(self, image):
        image = image.convert("RGB")
        image = np.array(image)
        image = image[:, :, ::-1].copy()   # RGB -> BGR
        return image

    @torch.no_grad()
    def _extract_single_image(self, image):
        raw_image = self._pil_to_bgr_numpy(image)
        raw_height, raw_width = raw_image.shape[:2]

        # Preprocessing
        transformed = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        transformed = torch.as_tensor(
            transformed.astype("float32").transpose(2, 0, 1),
            device=self.device
        )

        inputs = [{"image": transformed, "height": raw_height, "width": raw_width}]
        images = self.model.preprocess_image(inputs)

        # Backbone
        features = self.model.backbone(images.tensor)

        # RPN proposals
        proposals, _ = self.model.proposal_generator(images, features, None)

        # ROI head
        proposal_boxes = [x.proposal_boxes for x in proposals]
        roi_features = [features[f] for f in self.model.roi_heads.in_features]

        box_features = self.model.roi_heads._shared_roi_transform(
            roi_features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])   # [num_props, 2048]

        # Box predictor
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = \
            self.model.roi_heads.box_predictor(feature_pooled)

        outputs = FastRCNNOutputs(
            self.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.model.roi_heads.smooth_l1_beta,
        )

        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        # Attribute prediction
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # IMPORTANT: use transformed image size, not raw size
        image_shape = transformed.shape[1:]   # (H, W)

        instances, ids = None, None
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes,
                probs,
                image_shape,
                score_thresh=self.score_thresh,
                nms_thresh=nms_thresh,   # use loop variable
                topk_per_image=self.k,
            )
            if len(ids) == self.k:
                break

        if len(ids) < self.k:
            # ignore background column
            cls_probs = probs[:, :-1]                      # [R, C]
            max_scores, pred_classes = cls_probs.max(dim=1)  # [R], [R]
        
            # top-k proposals by best foreground score
            topk = min(self.k, max_scores.numel())
            ids = torch.topk(max_scores, k=topk, dim=0).indices
        
            # boxes is [R, C*4] -> reshape to [R, C, 4]
            num_classes = cls_probs.shape[1]
            boxes_per_class = boxes.view(-1, num_classes, 4)
        
            selected_classes = pred_classes[ids]                         # [k]
            selected_boxes = boxes_per_class[ids, selected_classes]      # [k, 4]
            selected_scores = max_scores[ids]                            # [k]
            selected_pred_classes = selected_classes                     # [k]
        
            from detectron2.structures import Instances, Boxes
            instances = Instances(image_shape)
            instances.pred_boxes = Boxes(selected_boxes)
            instances.scores = selected_scores
            instances.pred_classes = selected_pred_classes

        instances = detector_postprocess(instances, raw_height, raw_width)

        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()

        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        return instances, roi_features

    @torch.no_grad()
    def forward(self, *images):
        self.model.eval()

        batch_features = []
        batch_boxes = []

        for image in images:
            instances, features = self._extract_single_image(image)
            batch_features.append(features)
            batch_boxes.append(instances.pred_boxes.tensor)   # tensor, not Boxes object

        batch_features = torch.stack(batch_features, dim=0)   # [B, k, 2048]
        batch_boxes = torch.stack(batch_boxes, dim=0)         # [B, k, 4]

        return batch_features, batch_boxes
    

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
    

