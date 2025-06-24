from datasets import load_dataset
from cmcr.cmcr_model import HanCLIP, ModalityType, MCRType
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
from safetensors.torch import load_file
import numpy as np
from itertools import islice
import io
from Multilingual_CLIP.multilingual_clip import pt_multilingual_clip
import clip
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import login
import os

login(token="") 


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),  # CLIP 모델 크기
    transforms.ToTensor()
])


# Load the dataset
# streamed_dataset = load_dataset("kms7530/ko-coco-bal", split="validation", streaming=True, trust_remote_code=True)
streamed_dataset = load_dataset("jp1924/KoCC3M", split="validation", streaming=True, trust_remote_code=True)

samples = [x for x in tqdm(streamed_dataset)] 

# Load the model
text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")

hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()




images = []
# texts = []
for i in range(len(samples)):
    image = samples[i]['image'].convert("RGB")
    images.append(image)


print(f"Number of images: {len(images)}")

    
kor_text_embs_hanclip_full = []
img_embs_hanclip_full = []
all_images =[]


query_text = "누워있는 고양이"
top_k = 10

for i in range(0, len(images), 1024):
    images_batch = images[i:i+1024]
    if len(images_batch) == 0:
        continue  # 빈 배치는 스킵


    # Flatten the inputs
    flattened = {}
    # breakpoint()
    clip_vision_inputs = koclip_processor(images=images_batch, return_tensors="pt").to(device)
    kor_inputs = text_processor(query_text, return_tensors="pt", padding=True, truncation=True).to(device)
    for k, v in clip_vision_inputs.items():
        flattened[f"clip_vision_{k}"] = v
    for k, v in kor_inputs.items():
        flattened[f"kor_{k}"] = v
    # 임베딩 추출
    with torch.no_grad():
        outputs = hanclip.get_test_embeddings(flattened)
        kor_text_embs_hanclip, img_embs_hanclip = outputs[ModalityType.KOR_TEXT], outputs[ModalityType.VISION]


    img_embs_hanclip_full.append(img_embs_hanclip.cpu())
    kor_text_embs_hanclip_full.append(kor_text_embs_hanclip.cpu())
    all_images.extend(images_batch)  # 이미지 저장

# 텐서로 병합
img_embs_hanclip_full = torch.cat(img_embs_hanclip_full, dim=0)
kor_text_embs_hanclip_full = torch.cat(kor_text_embs_hanclip_full, dim=0)

# query 임베딩은 모두 동일하므로 하나만 사용
query_embedding = kor_text_embs_hanclip_full[0].unsqueeze(0)  # (1, D)
query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
img_embs = img_embs_hanclip_full / img_embs_hanclip_full.norm(dim=-1, keepdim=True)

# 코사인 유사도
similarities = torch.nn.functional.cosine_similarity(query_embedding, img_embs, dim=1)  # (N, )


# # Top-K 이미지 인덱스
# top_indices = torch.topk(similarities, k=top_k).indices.tolist()
# # fig, axes = plt.subplots(1, top_k, figsize=(20, 5))
# # for i, idx in enumerate(top_indices):
# #     axes[i].imshow(images[idx])
# #     axes[i].axis("off")
# #     axes[i].set_title(f"{i+1}")

# # plt.tight_layout()
# # plt.show()
# # plt.savefig(f"/home/aikusrv01/C-MCR/visualization/top_10_{query_text}.png")

    
save_dir = "/home/aikusrv01/C-MCR/visualization/retrieval_img"
os.makedirs(save_dir, exist_ok=True)

# Top-K 이미지 인덱스
top_indices = torch.topk(similarities, k=top_k).indices.tolist()

for i, idx in enumerate(top_indices):
    image = images[idx]
    filename = f"{i+1:02d}_{query_text}.png".replace(" ", "_")  # 파일명에 공백 제거
    image.save(os.path.join(save_dir, filename))
    print(f"Saved: {filename}")