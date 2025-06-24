import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn as nn
import os





# ======= 4️⃣ Semantic Enhancement =======
@torch.no_grad()
def semantic_enhancement(model, text_embedding_multi, text_embedding_clip, korean_emb, image_emb, temperature=0.01):
    """
    Step 1: Semantic Enhancement
    - English Text ↔ Image Memory
    - English Text ↔ Korean Memory
    """
    # print("🔸 Semantic Enhancement 수행 중...")

    text_embedding_multi = F.normalize(text_embedding_multi, dim=-1)
    korean_emb = F.normalize(korean_emb, dim=-1)
    similarity_image = torch.matmul(text_embedding_clip, image_emb.T) / temperature
    similarity_korean = torch.matmul(text_embedding_multi, korean_emb.T) / temperature
 
    # print(f"✅ Similarity Image Shape: {similarity_image.shape}")
    # print(f"✅ Similarity Korean Shape: {similarity_korean.shape}")
    weights_image = torch.softmax(similarity_image, dim=-1)
    weights_korean = torch.softmax(similarity_korean, dim=-1)

    # print(f"✅ Weights Image Shape: {weights_image.shape}")
    # print(f"✅ Weights Korean Shape: {weights_korean.shape}")
    enhanced_image = F.normalize(torch.matmul(weights_image, image_emb), dim=-1)
    enhanced_korean = F.normalize(torch.matmul(weights_korean, korean_emb), dim=-1)
    if model == "xlmr":
        enhanced_korean = text_embedding_multi
    # print(f"✅ Enhanced Image Shape: {enhanced_image.shape}")
    # print(f"✅ Enhanced Korean Shape: {enhanced_korean.shape}")
    return enhanced_korean, enhanced_image

if __name__ == "__main__":
    BATCH_SIZE = 256  # GPU 메모리 최적화
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print(f"Available GPUs: {torch.cuda.device_count()} GPUs detected.")

    target_text = " A woman riding a horse by the sea."

    print("모델 로딩 중...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    multilingual_model = AutoModel.from_pretrained("xlm-roberta-base").to(DEVICE)

    print("Embedding 로딩 중...")
    korean_emb = torch.load("/home/aikusrv01/C-MCR/korean_embedding.pt").to(DEVICE)
    image_emb = torch.load("/home/aikusrv01/C-MCR/image_embedding.pt").to(DEVICE)
    print(f"Korean Embedding Shape: {korean_emb.shape}")
    print(f"Image Embedding Shape: {image_emb.shape}")





    input_multi = tokenizer([target_text], return_tensors="pt", padding=True).to(DEVICE)
    inputs_clip = clip_processor(text=[target_text], return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        text_embedding_clip = clip_model.get_text_features(**inputs_clip)
        text_embedding_multi = multilingual_model(**input_multi).last_hidden_state[:, 0, :]

    print(f"✅ CLIP Text Embedding Shape: {text_embedding_clip.shape}")
    print(f"✅ Multi-lingual Text Embedding Shape (Before): {text_embedding_multi.shape}")
    enhanced_korean, enhanced_image = semantic_enhancement(text_embedding_clip, text_embedding_multi, korean_emb, image_emb)





