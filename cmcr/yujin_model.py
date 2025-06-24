import torch
import torch.nn.functional as F
from cmcr.yujin_projector import TriAlignHead
from cmcr.yujin_trunks import Trunk
from cmcr.type import ModalityType


TRIALIGN_CKPT = './checkpoints/trialign.pt'

class TriAlignModel:
    def __init__(self, device='cuda', load_ckpt=True) -> None:
        self.device = device
        self.trunk = Trunk(device)

        self.projector = TriAlignHead(
            text_input_dim=768,   # XLM-R
            image_input_dim=512,  # CLIP ViT-B/16
            proj_dim=512
        )

        if torch.cuda.is_available():
            self.projector = self.projector.to(device)

        if load_ckpt and TRIALIGN_CKPT and torch.exists(TRIALIGN_CKPT):
            print(f"✅ Loading checkpoint from {TRIALIGN_CKPT}")
            self.projector.load_state_dict(torch.load(TRIALIGN_CKPT, map_location=device))
        else:
            print("⚠️ No checkpoint loaded. Using randomly initialized projector.")
        
        self.projector.eval()

    @torch.no_grad()
    def get_text_embedding(self, input: dict):
        text_feat = self.trunk.get_text_feature(input[ModalityType.TEXT])
        proj_feat = self.projector.forward_text(text_feat)
        return F.normalize(proj_feat, dim=-1)

    @torch.no_grad()
    def get_vision_embedding(self, input: dict):
        image_feat = self.trunk.get_vision_feature(input[ModalityType.VISION])
        proj_feat = self.projector.forward_image(image_feat)
        return F.normalize(proj_feat, dim=-1)

    @torch.no_grad()
    def get_embeddings(self, input: dict):
        return {
            ModalityType.TEXT: self.get_text_embedding(input),
            ModalityType.VISION: self.get_vision_embedding(input)
        }
