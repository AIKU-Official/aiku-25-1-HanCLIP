from datasets import load_dataset
from cmcr.cmcr_model import HanCLIP, ModalityType, MCRType
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoProcessor, AutoModelForSeq2SeqLM
from PIL import Image
from safetensors.torch import load_file
import numpy as np
from itertools import islice
from io import BytesIO
from tqdm import tqdm
import requests
from torchvision import transforms
from huggingface_hub import login
import time
import statistics
login(token="") 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),  # CLIP 모델 크기
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

hanclip_image_times = []
hanclip_text_times = []

clip_translate_image_times = []
clip_translate_text_times = []
clip_translate_translate_times = []


def compute_recall_at_k(image_features, text_features, captions_per_image=5, k_list=[1, 5, 10]):
    """
    image_features: (N, D) tensor, normalized
    text_features: (N*C, D) tensor, normalized
    captions_per_image: how many captions per image (C)
    """
    num_images = image_features.shape[0]
    num_captions = text_features.shape[0]
    assert num_captions == num_images * captions_per_image

    # (N, N*C) similarity matrix
    sims = image_features @ text_features.T

    recalls_image_to_text = {f"Recall@{k}": 0.0 for k in k_list}

    for img_idx in range(num_images):
        # 정답 텍스트 인덱스들
        gt_caption_indices = list(range(img_idx * captions_per_image, (img_idx + 1) * captions_per_image))
        ranking = torch.argsort(sims[img_idx], descending=True)

        # rank는 top-k 안에 정답 캡션 중 하나라도 있으면 성공
        for k in k_list:
            top_k = ranking[:k].tolist()
            if any(gt in top_k for gt in gt_caption_indices):
                recalls_image_to_text[f"Recall@{k}"] += 1

    for k in k_list:
        recalls_image_to_text[f"Recall@{k}"] /= num_images

    # Text-to-Image: 각 캡션마다 정답 이미지가 1개
    sims_text_to_image = text_features @ image_features.T
    recalls_text_to_image = {f"Recall@{k}": 0.0 for k in k_list}

    for cap_idx in range(num_captions):
        gt_image_idx = cap_idx // captions_per_image
        ranking = torch.argsort(sims_text_to_image[cap_idx], descending=True)
        rank = (ranking == gt_image_idx).nonzero(as_tuple=True)[0].item() + 1

        for k in k_list:
            if rank <= k:
                recalls_text_to_image[f"Recall@{k}"] += 1

    for k in k_list:
        recalls_text_to_image[f"Recall@{k}"] /= num_captions

    return recalls_image_to_text, recalls_text_to_image

# Load the dataset
streamed_dataset = load_dataset("jp1924/KoCC3M", split="validation", streaming=True, trust_remote_code=True)
# samples = list(islice(streamed_dataset, 10000))
samples = [x for x in tqdm(islice(streamed_dataset, 5000))]
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
# Load the model
text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
# text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")

translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device).eval()

hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()

# koclip = AutoModel.from_pretrained("koclip/koclip-base-pt")
# koclip.to(device)
# koclip.eval()

tgt_lang = "eng_Latn"

images = []
texts = []
for i in range(len(samples)):
    image = samples[i]['image'].convert("RGB")
    text = samples[i]['caption']
    images.append(image)
    texts.extend(text)

print(f"Number of images: {len(images)}")
print(f"Number of texts: {len(texts)}")
    
kor_text_embs_hanclip_full = []
img_embs_hanclip_full = []
kor_text_embs_translate_full = []
eng_text_embs_translate_full = []
img_embs_clip_full = []
for i in range(0, len(images), 32):
    images_batch = images[i:i+32]
    if len(images_batch) == 0:
        continue  # 빈 배치는 스킵
    texts_batch = texts[5*i:5*i+len(images_batch)*5]
    if len(texts_batch) != len(images_batch) * 5:
        print(f"[Warning] texts_batch length mismatch at i={i}: got {len(texts_batch)} vs expected {len(images_batch)*5}")
        continue

    # for e5
    # texts_batch = ['query: ' + text for text in texts_batch]

    # Flatten the inputs
    flattened = {}
    # clip_vision_inputs = koclip_processor(images=images_batch, return_tensors="pt").to(device)
    clip_vision_inputs = clip_processor(images=images_batch, return_tensors="pt").to(device)
    kor_inputs = text_processor(texts_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    for k, v in clip_vision_inputs.items():
        flattened[f"clip_vision_{k}"] = v
    for k, v in kor_inputs.items():
        flattened[f"kor_{k}"] = v
    torch.cuda.synchronize()
    start_time = time.time()
    # 임베딩 추출
    with torch.no_grad():
        outputs = hanclip.get_test_embeddings(flattened)
        kor_text_embs_hanclip, img_embs_hanclip = outputs[ModalityType.KOR_TEXT], outputs[ModalityType.VISION]
    end_time = time.time()
    hanclip_text_times.append(end_time - start_time)
    # with torch.no_grad():
    #     kor_text_embs_koclip = koclip.get_text_features(**koclip_processor(text=texts_batch, return_tensors="pt", padding=True, truncation=True).to(device))
    #     img_embs_koclip = koclip.get_image_features(**koclip_processor(images=images_batch, return_tensors="pt").to(device))
    kor_text_embs_hanclip_full.append(kor_text_embs_hanclip.to('cpu'))
    img_embs_hanclip_full.append(img_embs_hanclip.to('cpu'))
    # kor_text_embs_koclip_full.append(kor_text_embs_koclip.to('cpu'))
    # img_embs_koclip_full.append(img_embs_koclip.to('cpu'))

for i in range(0, len(images), 32):
    images_batch = images[i:i+32]
    if len(images_batch) == 0:
        continue  # 빈 배치는 스킵
    texts_batch = texts[5*i:5*i+len(images_batch)*5]
    if len(texts_batch) != len(images_batch) * 5:
        print(f"[Warning] texts_batch length mismatch at i={i}: got {len(texts_batch)} vs expected {len(images_batch)*5}")
        continue
    # for e5
    # texts_batch = ['query: ' + text for text in texts_batch]

    # Flatten the inputs
    flattened = {}
    # clip_vision_inputs = koclip_processor(images=images_batch, return_tensors="pt").to(device)
    kor_inputs = translator_tokenizer(texts_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():    
        translated_tokens = translator_model.generate(
                **kor_inputs,
                forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=77,
        )
    translated = translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    end_time = time.time()
    clip_translate_translate_times.append(end_time - start_time)
    clip_vision_inputs = clip_processor(images=images_batch, return_tensors="pt").to(device)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        clip_img_embs = clip_model.get_image_features(**clip_vision_inputs)
    end_time = time.time()
    clip_translate_image_times.append(end_time - start_time)
    torch.cuda.synchronize()
    start_time = time.time()
    clip_text_inputs = clip_processor(text=translated, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
         clip_text_embs = clip_model.get_text_features(**clip_text_inputs)
    end_time = time.time()
    clip_translate_text_times.append(end_time - start_time)

    eng_text_embs_translate_full.append(clip_text_embs.cpu())
    img_embs_clip_full.append(clip_img_embs.cpu())

# Concatenate the results
kor_text_embs_hanclip_full = torch.cat(kor_text_embs_hanclip_full, dim=0).to(device)
img_embs_hanclip_full = torch.cat(img_embs_hanclip_full, dim=0).to(device)

eng_text_embs_translate_full = torch.cat(eng_text_embs_translate_full, dim=0).to(device)
img_embs_clip_full = torch.cat(img_embs_clip_full, dim=0).to(device)
# kor_text_embs_koclip_full = torch.cat(kor_text_embs_koclip_full, dim=0).to(device)
# img_embs_koclip_full = torch.cat(img_embs_koclip_full, dim=0).to(device)

print("HanCLIP Recall@1,5,10", compute_recall_at_k(img_embs_hanclip_full, kor_text_embs_hanclip_full))
print("Trnas + CLIP Recall@1,5,10", compute_recall_at_k(img_embs_clip_full,eng_text_embs_translate_full))
# print("KoCLIP Recall@1,5,10", compute_recall_at_k(img_embs_koclip_full, kor_text_embs_koclip_full))

# 결과 정리 출력
def print_stats(name, timings):
    print(f"{name} - 평균: {np.mean(timings):.4f}s, 표준편차: {np.std(timings):.4f}s")

print("\n--- Inference Time Statistics ---\n")

print_stats("HanCLIP Inference", hanclip_text_times)

print_stats("Translate+CLIP - 번역 시간", clip_translate_translate_times)
print_stats("Translate+CLIP - 텍스트 인코딩 시간", clip_translate_text_times)
print_stats("Translate+CLIP - 이미지 인코딩 시간", clip_translate_image_times)

total_time_hanclip = sum(hanclip_text_times)
total_time_translate_clip = (
    sum(clip_translate_translate_times) +
    sum(clip_translate_text_times) +
    sum(clip_translate_image_times)
)

print("\n--- Total Inference Time for Retrieval Task ---\n")
print(f"HanCLIP Total Time: {total_time_hanclip:.2f} seconds")
print(f"Translate + CLIP Total Time: {total_time_translate_clip:.2f} seconds")

print(f"\nAverage Time per Batch (HanCLIP): {total_time_hanclip / 32:.2f} seconds")
print(f"Average Time per Batch (Translate + CLIP): {total_time_translate_clip / 32:.2f} seconds")

