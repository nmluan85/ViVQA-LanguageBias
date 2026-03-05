import timm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from lavis.models.blip2_model.blip2_qformer import Blip2Qformer

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
    

