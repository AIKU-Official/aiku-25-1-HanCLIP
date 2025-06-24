import torch
import torch.nn as nn
import torch.nn.functional as F

class InterLoss(nn.Module):
    def __init__(self, temp=0.01):
        super(InterLoss, self).__init__()
        self.temp = temp
        
    def forward(self, kor_emb, eng_emb, text_emb, vision_emb):
        # kor_emb: (batch_size, 512)
        # eng_emb: (batch_size, 512)
        # text_emb: (batch_size, 512)
        # vision_emb: (batch_size, 512)
        batch_size = kor_emb.size(0)
        
        # similarity matrix
        sim_eng_text = torch.matmul(eng_emb, text_emb.T) / self.temp
        sim_text_eng = torch.matmul(text_emb, eng_emb.T) / self.temp
        
        sim_kor_vision = torch.matmul(kor_emb, vision_emb.T) / self.temp
        sim_vision_kor = torch.matmul(vision_emb, kor_emb.T) / self.temp

        labels = torch.arange(batch_size).to(kor_emb.device)

        loss_eng_text = (F.cross_entropy(sim_eng_text, labels) + F.cross_entropy(sim_text_eng, labels)) / 2
        loss_kor_vision = (F.cross_entropy(sim_kor_vision, labels) + F.cross_entropy(sim_vision_kor, labels)) / 2
        return loss_eng_text + loss_kor_vision


class IntraLoss(nn.Module):
    def __init__(self):
        super(IntraLoss, self).__init__()
        
    def forward(self, kor_emb, eng_emb, text_emb, vision_emb):
        # kor_emb: (batch_size, 512)
        # eng_emb: (batch_size, 512)
        # text_emb: (batch_size, 512)
        # vision_emb: (batch_size, 512)
        batch_size = kor_emb.size(0)
        
        dist_kor_eng = F.pairwise_distance(kor_emb, eng_emb, p=2) 
        dist_text_vision = F.pairwise_distance(text_emb, vision_emb, p=2)

        loss = (dist_kor_eng + dist_text_vision).mean() / 2
        return loss



if __name__ == "__main__":
    # Example usage
    kor_emb = F.normalize(torch.randn(32, 512), dim=-1)
    eng_emb = F.normalize(torch.randn(32, 512), dim=-1)
    text_emb = F.normalize(torch.randn(32, 512), dim=-1)
    vision_emb = F.normalize(torch.randn(32, 512), dim=-1)

    inter_loss_fn = InterLoss(temp=0.1)
    intra_loss_fn = IntraLoss()

    inter_loss = inter_loss_fn(kor_emb, eng_emb, text_emb, vision_emb)
    intra_loss = intra_loss_fn(kor_emb, eng_emb, text_emb, vision_emb)

    print("Inter Loss:", inter_loss.item())
    print("Intra Loss:", intra_loss.item())

    