# 프로젝트명

📢 2025년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다
🎉 2025년 1학기 AIKU Conference 열심히상 수상!

## 소개

> **HanCLIP**: A lightweight and efficient cross-lingual vision-language model for Korean, built on top of CLIP

**HanCLIP**은 대규모 한국어–이미지 쌍 없이도 한국어와 시각 정보 간 의미 정렬을 가능하게 하는 효율적인 비전-언어 모델입니다. 

영어를 의미적 중간 다리(semantic pivot)로 활용하여, 기존 CLIP의 강력한 시각 표현 능력을 유지하면서도 한국어 질의에 대한 이미지 검색 및 분류가 가능합니다.

## 방법론

HanCLIP은 이미지–한국어 간 직접적인 정렬 없이, **영어를 의미적 매개체(semantic pivot)**로 활용하여 이미지–한국어 표현 간 **공유 임베딩 공간(shared embedding space)**을 학습하는 구조입니다.

![Overall Architecture](asset/pipeline.png)

### Architecture Components

1. **Frozen Image Encoder**: CLIP ViT-B/32  
2. **Frozen Multilingual Text Encoder**: MiniLM-L12  
3. **Trainable Projection Heads**: f₁(·), f₂(·)

---

### Input 구성

- **영어 Text Query**: *eᵢ*
  - *eᵢᴵ*: CLIP Text Encoder로부터 추출된 영어 임베딩
  - *eᵢᴷ*: Multilingual Text Encoder로부터 추출된 영어 임베딩

- **이미지 메모리**: *V = {v₁, ..., vₙ}*  
  - CLIP Image Encoder로 전처리된 이미지 임베딩

- **한국어 텍스트 메모리**: *K = {k₁, ..., kₘ}*  
  - Multilingual Encoder로 전처리된 한국어 임베딩

---

### Semantic Enhancement

- 이미지 쿼리 및 한국어 쿼리로부터 **pseudo image** *vᵢᴵ*, **pseudo Korean text** *kᵢᴷ* 생성  
- Softmax-weighted aggregation으로 유사한 메모리에서 표현 보완

![Cross-modal and Cross-lingual Semantic Enhancement](asset/semantic_enhancement.png)

- 각 임베딩에 **노이즈 추가 및 L2 정규화** 수행:

![Perturbed Embedding Semantic Enhancement](asset/noise_perturb.png)

---

### Inter-alignment Loss

- 4가지 임베딩을 각각 projection head에 통과:

```
êᵢᴵ = f₁(~eᵢᴵ)  
êᵢᴷ = f₂(~eᵢᴷ)  
v̂ᵢᴵ = f₁(~vᵢᴵ)  
k̂ᵢᴷ = f₂(~kᵢᴷ)
```

- 두 쌍 (image-text)의 **symmetric contrastive loss** 적용

![Text-text Inter-alignment Loss](asset/text_inter.png)  
![pseudo image-Korean Inter-alignment Loss](asset/pseudo_inter.png)

- 최종 Inter-alignment Loss

```
𝓛 = 𝓛_text + 𝓛_pseudo
```
---

### Intra-alignment Loss

- **모달리티 간 격차(modality gap)**를 줄이기 위해 attractive term만 사용  
- 동일 의미의 쌍 간 거리 최소화

![Intra-alignment Loss](asset/intra.png)

---

### Total Loss

```
𝓛 = 𝓛_inter + λ × 𝓛_intra
```

---

### Training Details

- 모든 인코더는 **Frozen**
- **Projection Head**만 학습
- 학습 시 Gaussian noise + normalization → generalization 및 표현력 향상

---


## 환경 설정
environment를 생성하고 아래 코드를 실행하여 dependency들을 설치

  ```
    conda create -n env_name python=3.9
    conda activate env_name
    pip install -r requirements.txt
  ```

## 사용 방법
### 1. Quantitative retrieval result (Recall@K)
  ```
    # MSCOCO-Korean 데이터셋에 대한 retrieval result
    python retrieval_quan_cc3m.py

    # KoCC3M 데이터셋에 대한 retrieval result
    python retrieval_quan_mscoco.py
  ```
  - 실행 시 hanclip, multilingual_clip, koclip에 대한 결과 metric 표시

### 2. Quantitative classification result (F1 score)
  ```
    sh classification.sh
  ```
  - 위 파일에서 --dataset을 cifar10, stl10, caltech101, cifar100 으로 바꿔서 실행
  - 위 파일에서 --model을 hanclip, multilingual_clip, koclip 으로 바꿔서 실행

### 3. Qualitative retrieval result
  ```
    python retrieval_img.py
  ```
  - 위 파일에서 query_text를 원하는 문장으로 수정가능
  - 해당 query_text에 대한 top10 이미지 retrieval 결과가 {idx}_{query_text}.png 형식으로 저장됨

### 4. Inference time 
  ```
    python inference_time.py
  ```

### 5. Visualization of cosine simalarity
  ```
    python visualization_cossim.py
  ```
  - 실행 시 hanclip, multilingual_clip, koclip에 대한 결과가 각각 cosine_similarity_{model_name}.png 로 저장됨

### 6. Visualization of image embedding
  ```
    python visualization_imgemb.py
  ```
  - 실행 시 hanclip, clip, koclip에 대한 결과가 각각 cifar10_umap_{model_name}.png 로 저장됨

## 예시 결과
### 1. Quantitative retrieval result (Recall@K)
![Retrieval Result](asset/retrieval_quan.png)

### 2. Quantitative classification result (F1 score)
![Classification Result](asset/classification.png)

### 3. Qualitative retrieval result
![Retrieval Result Image](asset/qualitative_result.png)

### 4. Inference time 
![Inference Time Result](asset/inference_time.png)

### 5. Visualization of cosine simalarity
![Cosine Similarity Visualization](asset/cosine_similarity_result.png)

### 6. Visualization of image embedding
![Image Embedding Visualization](asset/image_emb_result.png)

## 팀원

  | 팀원                            | 역할                                       |
| ----------------------------- | ---------------------------------------- |
| [문승기](https://github.com/moon44432) |    Dataset curation, Architecture figure, Evaluation   |
| [성유진](https://github.com/dinyudin203)      |    Dataset curation, Train, Evaluation    |
| [신동현](https://github.com/Donghyun1228)     |    Dataset curation, Train(Overall), Evaluation    |
| [정다현](https://github.com/dhyun22)        |    Dataset curation, Paper writing, Analysis    |
