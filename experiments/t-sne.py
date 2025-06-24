from datasets import load_dataset
from cmcr.cmcr_model import HanCLIP, ModalityType, MCRType
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
from safetensors.torch import load_file
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import islice
import io

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# minilm
text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")

# Load the dataset
streamed_dataset = load_dataset("kms7530/ko-coco-bal", split="validation", streaming=True, trust_remote_code=True)
samples = list(islice(streamed_dataset, 10))
# Load the model
hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()

koclip = AutoModel.from_pretrained("koclip/koclip-base-pt")
koclip.to(device)
koclip.eval()


images = []
texts = []
for i in range(len(samples)):
    image = samples[i]['image'].convert("RGB")
    text = samples[i]['caption_ko']
    images.append(image)
    texts.append(text)
    
flattened = {}
clip_vision_inputs = koclip_processor(images=images, return_tensors="pt").to(device)
kor_inputs = text_processor(texts, return_tensors="pt", padding=True, truncation=True).to(device)

for k, v in clip_vision_inputs.items():
    flattened[f"clip_vision_{k}"] = v
for k, v in kor_inputs.items():
    flattened[f"kor_{k}"] = v

# 임베딩 추출
with torch.no_grad():
    outputs = hanclip.get_test_embeddings(flattened)
    kor_text_embs_hanclip, img_embs_hanclip = outputs[ModalityType.KOR_TEXT], outputs[ModalityType.VISION]

with torch.no_grad():
    kor_text_embs_koclip = koclip.get_text_features(**koclip_processor(texts, return_tensors="pt", padding=True, truncation=True).to(device))
    img_embs_koclip = koclip.get_image_features(**koclip_processor(images=images, return_tensors="pt").to(device))

# 텐서를 numpy로 변환
kor_text_np_hanclip = kor_text_embs_hanclip.cpu().numpy()
img_np_hanclip = img_embs_hanclip.cpu().numpy()

kor_text_np_koclip = kor_text_embs_koclip.cpu().numpy()
img_np_koclip = img_embs_koclip.cpu().numpy()

# 전체 임베딩 합치기
all_embs_hanclip = np.concatenate([kor_text_np_hanclip, img_np_hanclip], axis=0)  # shape: (2N, dim)
all_embs_koclip = np.concatenate([kor_text_np_koclip, img_np_koclip], axis=0)  # shape: (2N, dim)


# t-SNE 실행 (2N개 입력)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result_hanclip = tsne.fit_transform(all_embs_hanclip)  # shape: (2N, 2)
tsne_result_koclip = tsne.fit_transform(all_embs_koclip)  # shape: (2N, 2)

# 시각화
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

def plot_tsne(subplt, tsne_result, title):
    text_points = tsne_result[:len(kor_text_np_hanclip)]
    image_points = tsne_result[len(kor_text_np_hanclip):]

    # 포인트 그리기
    subplt.scatter(text_points[:, 0], text_points[:, 1], c='blue', label='Text', alpha=0.7)
    subplt.scatter(image_points[:, 0], image_points[:, 1], c='red', label='Image', alpha=0.7)

    # 쌍 단위로 선 긋기
    for i in range(len(kor_text_np_hanclip)):
        x_coords = [text_points[i, 0], image_points[i, 0]]
        y_coords = [text_points[i, 1], image_points[i, 1]]
        subplt.plot(x_coords, y_coords, c='gray', linestyle='--', alpha=0.5)

    # 그래프 설정
    subplt.legend()
    subplt.set_title(title)
    subplt.set_xlabel("t-SNE dim 1")
    subplt.set_ylabel("t-SNE dim 2")
    subplt.grid(True)

plot_tsne(axs[0], tsne_result_hanclip, "t-SNE Visualization - HanCLIP")
plot_tsne(axs[1], tsne_result_koclip, "t-SNE Visualization - KoCLIP")

plt.savefig("tsne_output.png")  # show 대신 저장
print("t-SNE plot saved to tsne_output.png")
