import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from cmcr.cmcr_projector import HanCLIP_Head
from cmcr.trunks import Trunk

from cmcr.type import ModalityType, MCRType
from cmcr.extract_embedding import semantic_enhancement

from cmcr.loss import InterLoss, IntraLoss

inter_loss = InterLoss()
intra_loss = IntraLoss()


# CLAP_CLIP = 'checkpoints/clap_clip.pt'
# ULIP_CLIP = 'checkpoints/ulip_clip.pt'

# class C_MCR_CLAPCLIP():
#     def __init__(self, device='cpu') -> None:
#         super().__init__()
#         self.device = device
        
#         self.trunk = Trunk(device) # dict
#         self.cmcr_head = CLAPCLIP_Head()
#         self.cmcr_head.load_state_dict(torch.load(CLAP_CLIP, map_location='cpu'))
#         self.cmcr_head.to(device)
#         self.cmcr_head.eval()
    
#     @torch.no_grad()
#     def project_features(self, features: dict) -> dict:
#         cmcr_embeddings = {}
#         cmcr_embeddings[ModalityType.VISION] = self.project_clip(features[ModalityType.VISION])
#         cmcr_embeddings[ModalityType.TEXT]   = self.project_clip(features[ModalityType.TEXT])
#         cmcr_embeddings[ModalityType.AUDIO]  = self.project_clap(features[ModalityType.AUDIO])

#         cmcr_embeddings[ModalityType.VISION] = F.normalize(cmcr_embeddings[ModalityType.VISION], dim=-1)
#         cmcr_embeddings[ModalityType.TEXT]   = F.normalize(cmcr_embeddings[ModalityType.TEXT], dim=-1)
#         cmcr_embeddings[ModalityType.AUDIO]  = F.normalize(cmcr_embeddings[ModalityType.AUDIO], dim=-1)
        
#         return cmcr_embeddings
    
#     @torch.no_grad()
#     def project_clap(self, clap_emb: Tensor) -> Tensor:
#         return self.cmcr_head.Head_A(clap_emb)
    
#     @torch.no_grad()
#     def project_clip(self, clip_emb: Tensor) -> Tensor:
#         return self.cmcr_head.Head_B(clip_emb)
    
#     @torch.no_grad()
#     def get_embeddings(self, input: dict) -> dict:
#         features = {}
#         features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
#         features[ModalityType.TEXT]   = self.trunk.get_text_feature(input[ModalityType.TEXT])
#         features[ModalityType.AUDIO]  = self.trunk.get_audio_feature(input[ModalityType.AUDIO])
#         cmcr_embeddings = self.project_features(features)
#         return cmcr_embeddings
    
#     @torch.no_grad()
#     def get_vision_embedding(self, input: dict) -> Tensor:
#         features = self.trunk.get_vision_feature(input[ModalityType.VISION])
#         features = self.project_clip(features)
#         return F.normalize(features, dim=-1)
    
#     @torch.no_grad()
#     def get_text_embedding(self, input: dict) -> Tensor:
#         features = self.trunk.get_text_feature(input[ModalityType.TEXT])
#         features = self.project_clip(features)
#         return F.normalize(features, dim=-1)
    
#     @torch.no_grad()
#     def get_audio_embedding(self, input: dict) -> Tensor:
#         features = self.trunk.get_audio_feature(input[ModalityType.AUDIO])
#         features = self.project_clap(features)
#         return F.normalize(features, dim=-1)


# class C_MCR_ULIPCLIP():
#     def __init__(self, device='cpu') -> None:
#         super().__init__()
#         self.device = device

#         self.trunk = Trunk(device) # dict
#         self.cmcr_head = ULIPCLIP_Head()
#         self.cmcr_head.load_state_dict(torch.load(ULIP_CLIP, map_location='cpu'))
#         self.cmcr_head.to(device)
#         self.cmcr_head.eval()
    
#     @torch.no_grad()
#     def project_features(self, features: dict) -> dict:
#         cmcr_embeddings = {}
#         cmcr_embeddings[ModalityType.VISION] = self.project_clip(features[ModalityType.VISION])
#         cmcr_embeddings[ModalityType.TEXT]   = self.project_clip(features[ModalityType.TEXT])
#         cmcr_embeddings[ModalityType.PC]     = self.project_ulip(features[ModalityType.PC])

#         cmcr_embeddings[ModalityType.VISION] = F.normalize(cmcr_embeddings[ModalityType.VISION], dim=-1)
#         cmcr_embeddings[ModalityType.TEXT]   = F.normalize(cmcr_embeddings[ModalityType.TEXT], dim=-1)
#         cmcr_embeddings[ModalityType.PC]     = F.normalize(cmcr_embeddings[ModalityType.PC], dim=-1)
        
#         return cmcr_embeddings
    
#     @torch.no_grad()
#     def project_ulip(self, ulip_emb: Tensor) -> Tensor:
#         return self.cmcr_head.Head_B(ulip_emb)
    
#     @torch.no_grad()
#     def project_clip(self, clip_emb: Tensor) -> Tensor:
#         return self.cmcr_head.Head_A(clip_emb)
    
#     @torch.no_grad()
#     def get_embeddings(self, input: dict) -> dict:
#         features = {}
#         features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
#         features[ModalityType.TEXT]   = self.trunk.get_text_feature(input[ModalityType.TEXT])
#         features[ModalityType.PC]     = self.trunk.get_3d_feature(input[ModalityType.PC])
#         cmcr_embeddings = self.project_features(features)
#         return cmcr_embeddings
    
#     @torch.no_grad()
#     def get_vision_embedding(self, input: dict) -> Tensor:
#         features = self.trunk.get_vision_feature(input[ModalityType.VISION])
#         features = self.project_clip(features)
#         return F.normalize(features, dim=-1)
    
#     @torch.no_grad()
#     def get_text_embedding(self, input: dict) -> Tensor:
#         features = self.trunk.get_text_feature(input[ModalityType.TEXT])
#         features = self.project_clip(features)
#         return F.normalize(features, dim=-1)
    
#     @torch.no_grad()
#     def get_3d_embedding(self, input: dict) -> Tensor:
#         features = self.trunk.get_3d_feature(input[ModalityType.PC])
#         features = self.project_ulip(features)
#         return F.normalize(features, dim=-1)
    
class HanCLIP(nn.Module):
    def __init__(self, model, kor_memory_path, image_memory_path, device='cpu', noise=True) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.noise = noise

        self.register_buffer("kor_memory", torch.load(kor_memory_path))
        self.register_buffer("image_memory", torch.load(image_memory_path))

        self.trunk = Trunk(device, model)
        self.cmcr_head = HanCLIP_Head(model)

        for param in self.trunk.parameters():
                param.requires_grad = False

    def loss_fn(self, kor_emb, eng_emb, text_emb, vision_emb, lamb = 0.1):
        loss = inter_loss(kor_emb, eng_emb, text_emb, vision_emb) + lamb * intra_loss(kor_emb, eng_emb, text_emb, vision_emb)
        return loss.unsqueeze(0)

    def project_features(self, features: dict) -> dict:
        cmcr_embeddings = {}
        for modality in features.keys():
            if modality == ModalityType.VISION:
                cmcr_embeddings[modality] = self.project_clip(features[modality])
            elif modality == ModalityType.TEXT:
                cmcr_embeddings[modality] = self.project_clip(features[modality])
            elif modality == ModalityType.KOR_TEXT:
                cmcr_embeddings[modality] = self.project_text(features[modality])
            elif modality == ModalityType.ENG_TEXT:
                cmcr_embeddings[modality] = self.project_text(features[modality])
        
        return cmcr_embeddings
    
    def project_text(self, text_emb: Tensor) -> Tensor:
        return F.normalize(self.cmcr_head.Head_A(text_emb), dim=-1)
    
    def project_clip(self, clip_emb: Tensor) -> Tensor:
        return F.normalize(self.cmcr_head.Head_B(clip_emb), dim=-1)


    def get_embeddings(self, input: dict) -> dict:
        input[ModalityType.VISION] = {
            'pixel_values': input["clip_vision_pixel_values"],
        }
        input[ModalityType.TEXT] = {
            'input_ids': input["clip_text_input_ids"],
            'attention_mask': input["clip_text_attention_mask"],
        }
        input[ModalityType.ENG_TEXT] = {
            'input_ids': input["eng_input_ids"],
            'attention_mask': input["eng_attention_mask"]
        }
        input[ModalityType.KOR_TEXT] = {
            'input_ids': input["kor_input_ids"],
            'attention_mask': input["kor_attention_mask"]
        }
        features = {}
        features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features[ModalityType.TEXT]   = self.trunk.get_text_feature(input[ModalityType.TEXT])

        features[ModalityType.ENG_TEXT] = self.trunk.get_eng_text_feature(input[ModalityType.ENG_TEXT])
        features[ModalityType.KOR_TEXT] = self.trunk.get_kor_text_feature(input[ModalityType.KOR_TEXT])
        features = self.project_features(features)
        return features

    def get_test_embeddings(self, input: dict) -> dict:
        input[ModalityType.VISION] = {
            'pixel_values': input["clip_vision_pixel_values"],
        }
        input[ModalityType.KOR_TEXT] = {
            'input_ids': input["kor_input_ids"],
            'attention_mask': input["kor_attention_mask"]
        }
        features = {}
        features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features[ModalityType.KOR_TEXT] = self.trunk.get_kor_text_feature(input[ModalityType.KOR_TEXT])
        features = self.project_features(features)
        return features
    
    def forward(self, **input) -> Tensor:
        input[ModalityType.TEXT] = {
            'input_ids': input.get("clip_input_ids"),
            'attention_mask': input.get("clip_attention_mask"),
        }
        input[ModalityType.ENG_TEXT] = {
            'input_ids': input.get("eng_input_ids"),
            'attention_mask': input.get("eng_attention_mask")
        }
        features = {}
        features[ModalityType.ENG_TEXT] = self.trunk.get_eng_text_feature(input[ModalityType.ENG_TEXT])
        features[ModalityType.TEXT] = self.trunk.get_text_feature(input[ModalityType.TEXT])

        features[ModalityType.KOR_TEXT], features[ModalityType.VISION] = semantic_enhancement(
            self.model,
            features[ModalityType.ENG_TEXT], 
            features[ModalityType.TEXT], 
            self.kor_memory, 
            self.image_memory, 
            temperature=0.01)

        # 노이즈 더하기
        if self.noise and self.model != 'xlmr':
            for key, tensor in features.items():
                noise = torch.randn_like(tensor) * 0.025 # 이전에는 0.06, 0.0424
                features[key] = F.normalize(tensor + noise, dim=-1)

        features = self.project_features(features)
        
        loss = self.loss_fn(
            features[ModalityType.KOR_TEXT], 
            features[ModalityType.ENG_TEXT], 
            features[ModalityType.TEXT], 
            features[ModalityType.VISION]
        )
        return {'loss': loss, **features}

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HanCLIP(device=device)
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)