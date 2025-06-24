import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from PIL import Image
import matplotlib.pyplot as plt
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======= 1️⃣ 최종 임베딩 로딩 =======
print("🔹 임베딩 로딩 중...")
completed_korean = torch.load("/home/aikusrv01/C-MCR/completed_korean.pt").to(DEVICE)
completed_image = torch.load("/home/aikusrv01/C-MCR/completed_image.pt").to(DEVICE)

print(f"✅ Korean Embedding Shape: {completed_korean.shape}")
print(f"✅ Image Embedding Shape: {completed_image.shape}")

# ✅ Query는 1개의 문장에 대한 임베딩이어야 함
if completed_korean.dim() > 1 and completed_korean.size(0) > 1:
    print("🔹 Query 임베딩의 첫 번째 벡터만 사용합니다.")
    completed_korean = completed_korean[0].unsqueeze(0)

print(f"✅ Query Shape (Korean): {completed_korean.shape}")

# ======= 2️⃣ Embedding 로딩 =======
print("🔹 Embedding 로딩 중...")
korean_emb = torch.load("/home/aikusrv01/C-MCR/korean_embedding.pt").to(DEVICE)
image_emb = torch.load("/home/aikusrv01/C-MCR/image_embedding.pt").to(DEVICE)

# 🔸 🟢 Transpose 제대로 수행하기
if korean_emb.shape[0] == 768:  
    print("🔸 Transpose 진행 중...")
    korean_emb = korean_emb.transpose(0, 1).contiguous()  # (768, 616767) → (616767, 768)
    print(f"✅ Transpose 후 Shape: {korean_emb.shape}")

# 🔹 Korean Embedding 차원 조정
print("🔹 Korean Embedding 차원 조정 중...")
project_korean = nn.Linear(768, 512).to(DEVICE)
korean_emb = project_korean(korean_emb)
print(f"✅ Korean Embedding Shape after Projection: {korean_emb.shape}")
print(f"✅ Image Embedding Shape: {image_emb.shape}")

# 🔹 전체 Korean Memory 로딩 중...
print("🔹 전체 Korean Memory 로딩 중...")
all_korean_emb = torch.load("/home/aikusrv01/C-MCR/korean_embedding.pt").to(DEVICE)

# 🔸 메모리에서 Transpose를 확실히 반영
if all_korean_emb.shape[0] == 768:
    print("🔸 Transpose 진행 중...")
    all_korean_emb = all_korean_emb.transpose(0, 1).contiguous()
    print(f"✅ Transpose 후 Shape: {all_korean_emb.shape}")

# 🔹 Korean Memory 차원 조정
print("🔹 Korean Memory 차원 조정 중...")
all_korean_emb = project_korean(all_korean_emb)
print(f"✅ 투영 완료! New Shape: {all_korean_emb.shape}")

# 🔸 Korean Memory 저장
torch.save(all_korean_emb, "/home/aikusrv01/C-MCR/completed_korean.pt")
print(f"✅ 전체 Korean Memory 저장 완료! (Shape: {all_korean_emb.shape})")

top_k = 5
# ======= 3️⃣ Text → Korean Text 검색 =======
print("\n🔍 Text → Korean Text 검색 중...")

# 🔹 Korean Memory 로딩
korean_memory = torch.load("/home/aikusrv01/C-MCR/completed_korean.pt").to(DEVICE)

# 🔹 유사도 계산 (Cosine Similarity)
similarity_korean = F.cosine_similarity(completed_korean, korean_memory)

# 🔹 상위 5개 검색
values_ko, indices_ko = torch.topk(similarity_korean, top_k, largest=True)

# 🔹 결과 출력
print("\n🔹 [Text → Korean Text] 검색 결과 (Top 5):")
for rank, (score, idx) in enumerate(zip(values_ko, indices_ko)):
    print(f"{rank + 1}. Korean Text Index: {idx.item()}, Similarity: {score.item()}")

# ======= 4️⃣ 실제 한국어 문장 로딩 및 매칭 =======
korean_texts_path = "/home/aikusrv01/C-MCR/datasets/MSCOCO_korean/MSCOCO_train_val_Korean.json"

print("\n🔍 실제 한국어 문장 확인 중...")
with open(korean_texts_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔸 JSON의 길이 확인
max_index = len(data)
print(f"✅ JSON 데이터 개수: {max_index}")

# 🔹 상위 5개의 인덱스에 해당하는 한국어 문장 출력
for rank, idx in enumerate(indices_ko):
    if idx.item() >= max_index:
        print(f"⚠️ 경고: Index {idx.item()}가 JSON 길이를 초과합니다. 건너뜁니다.")
        continue
    print(f"{rank + 1}. Index {idx.item()} → Korean Text: {data[idx.item()]['caption_ko']}")
