
import argparse
import torch
import json
import os
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, help="이미지는 폴더 경로(쉼표로 다중 지정 가능), 텍스트는 파일 경로", default=None)
parser.add_argument("--type", type=str, choices=["image", "korean"], default=None)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=16, help="배치 크기 (기본값: 16)")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_image(fpath):
    try:
        image = Image.open(fpath).convert("RGB")
        return image
    except Exception as e:
        print(f"❌ 이미지 열기 실패: {fpath}, 오류: {e}")
        return None

@torch.no_grad()
# TODO: refactor get_vision_feature
def get_vision_feature():
    print(f"🔍 모델 로딩 중...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_features = []

    if isinstance(args.source_path, str):
        args.source_path = [p.strip() for p in args.source_path.split(",")]

    all_files = []
    for path in args.source_path:
        print(f"📂 탐색 중: {path}")
        if not os.path.exists(path):
            print(f"❌ 경로가 존재하지 않습니다: {path}")
            continue
        
        with os.scandir(path) as it:
            all_files.extend([entry.path for entry in it if entry.is_file() and 'train' in entry.name])

    print(f"✅ 로딩된 파일 개수: {len(all_files)}")

    if len(all_files) == 0:
        print("⚠️ 로딩된 이미지가 없습니다. 경로와 파일 형식을 확인하세요.")
        return None

    print(f"🚀 이미지 로딩 중 (멀티프로세싱)...")
    all_images = []
    with ProcessPoolExecutor() as executor:
        for image in tqdm(executor.map(load_image, all_files), total=len(all_files), desc="📥 이미지 로딩 중"):
            if image is not None:
                all_images.append(image)

    print(f"✅ 로딩된 이미지 개수: {len(all_images)}")

    
    for i in tqdm(range(0, len(all_images), args.batch_size), desc="📊 이미지 임베딩 중 (GPU 사용 중)"):
        batch = all_images[i:i + args.batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        features = model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1)
        image_features.append(features)
    
    image_embeddings = torch.cat(image_features, dim=0)
    print(f'🔍 임베딩 완료 - Original Image Count: {len(all_images)}, Embedding Size: {image_embeddings.size()}')
    return image_embeddings

@torch.no_grad()
def get_korean_feature():
    print(f"🔍 모델 로딩 중...")
    if args.model == "e5":
        model = SentenceTransformer(E5_MODEL_PATH).to(device)
    elif args.model == "minilm":
        model = SentenceTransformer(MINILM_MODEL_PATH).to(device)
    
    print(f"📂 JSON 파일 탐색 중: {args.source_path}")
    
    all_captions = []


    with open(args.source_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # <== 여기서 전체를 로딩합니다.

    # 각 항목에서 caption_ko를 추출합니다.
    for obj in data:
        if "caption_ko" in obj and isinstance(obj["caption_ko"], list):
            valid_captions = [caption for caption in obj['caption_ko'] if isinstance(caption, str)]
            all_captions.extend(valid_captions)

    print(all_captions[:10])  # 로딩된 캡션의 일부를 출력합니다.

    if args.model == "e5":
        all_captions = ["query: " + caption for caption in all_captions]

    print(f"✅ 로딩된 캡션 개수: {len(all_captions)}")
    if len(all_captions) == 0:
        print("⚠️ 로딩된 텍스트가 없습니다. JSON 구조를 확인하세요.")
        return None

    # ======= 임베딩 추출 =======
    print(f"🚀 텍스트 임베딩 추출 중...")
    all_embeddings = []

    for i in tqdm(range(0, len(all_captions), args.batch_size), desc="📊 텍스트 임베딩 중"):
        batch = all_captions[i:i + args.batch_size]
        outputs = model.encode(batch, device=device, convert_to_tensor=True, normalize_embeddings=True)
        all_embeddings.append(outputs)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f'🔍 임베딩 완료 - Original Text Count: {len(all_captions)}, Embedding Size: {all_embeddings.size()}, Embedding Length: {torch.norm(all_embeddings[0], p=2)}')
    return all_embeddings
# ======= Main Execution =======
if args.type == "image":
    image_embedding = get_vision_feature()
    if image_embedding is not None:
        torch.save(image_embedding, "image_embedding.pt")
        print("✅ 이미지 임베딩 저장 완료: image_embedding.pt")
elif args.type == "korean":
    text_embedding = get_korean_feature()
    if text_embedding is not None:
        torch.save(text_embedding, "text_embedding.pt")
        print("✅ 텍스트 임베딩 저장 완료: text_embedding.pt")
else:
    print("❌ 잘못된 타입입니다. --type 인자는 'image' 또는 'korean'이어야 합니다.")
