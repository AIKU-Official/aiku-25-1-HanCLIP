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
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

text_processor = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
koclip_processor = AutoProcessor.from_pretrained("koclip/koclip-base-pt")


koclip = AutoModel.from_pretrained("koclip/koclip-base-pt")
koclip.to(device)
koclip.eval()


# Load the dataset
streamed_dataset = load_dataset("kms7530/ko-coco-bal", split="validation", streaming=True, trust_remote_code=True)
samples = list(islice(streamed_dataset, 50))
# Load the model
hanclip = HanCLIP('minilm', "korean_embedding_MiniLM_MSCOCO.pt", "image_embedding_CLIP_imagenet.pt")
hanclip_checkpoint = load_file("results/minilm_noise_final/checkpoint-17600/model.safetensors")
hanclip.load_state_dict(hanclip_checkpoint, strict=False)
hanclip.to(device)
hanclip.eval()


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


with torch.no_grad():
    # projector 통과 전
    clip_input = {
        'pixel_values': flattened["clip_vision_pixel_values"]
    }
    kor_input = {
        'input_ids': flattened["kor_input_ids"],
        'attention_mask': flattened["kor_attention_mask"]
    }

    img_pre = hanclip.trunk.get_vision_feature(clip_input)  # shape: (N, D1)
    text_pre = hanclip.trunk.get_kor_text_feature(kor_input)  # shape: (N, D2)


    # projector 통과 후
    outputs_post = hanclip.get_test_embeddings(flattened)
    img_post = outputs_post[ModalityType.VISION]  # shape: (N, d)
    text_post = outputs_post[ModalityType.KOR_TEXT]  # shape: (N, d)

    #Koclip
    kor_text_embs_koclip = koclip.get_text_features(**koclip_processor(texts, return_tensors="pt", padding=True, truncation=True).to(device))
    img_embs_koclip = koclip.get_image_features(**koclip_processor(images=images, return_tensors="pt").to(device))
    koclip_img = F.normalize(img_embs_koclip, dim=-1).cpu().numpy()
    koclip_txt = F.normalize(kor_text_embs_koclip, dim=-1).cpu().numpy()
    ####################cosine sim#########################
    # normalize pre for fair comparison (optional)
    img_pre = F.normalize(img_pre, dim=-1)
    text_pre = F.normalize(text_pre, dim=-1)
    
    pca = PCA(n_components=50)
    img_pre_reduced = pca.fit_transform(img_pre.cpu().numpy())
    text_pre_reduced = pca.fit_transform(text_pre.cpu().numpy())

    # t-SNE separately (because of dim mismatch)
    tsne_img_pre = TSNE(n_components=2, perplexity=5).fit_transform(img_pre.cpu().numpy())
    tsne_txt_pre = TSNE(n_components=2, perplexity=5).fit_transform(text_pre.cpu().numpy())
    tsne_post = TSNE(n_components=2, perplexity=5).fit_transform(
        torch.cat([text_post, img_post], dim=0).cpu().numpy()
    )
    tsne_koclip = TSNE(n_components=2, perplexity=5).fit_transform(np.concatenate([koclip_txt, koclip_img], axis=0))




    # 시각화
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    # before projector: 분리 시각화
    # axs[0].scatter(tsne_txt_pre[:, 0], tsne_txt_pre[:, 1], label='Text (Pre)', color='blue')
    # axs[0].scatter(tsne_img_pre[:, 0], tsne_img_pre[:, 1], label='Image (Pre)', color='red')
    # axs[0].legend()
    # axs[0].set_title("Before Projector (Separate Spaces)")
    # axs[0].grid(True)
    axs[0].scatter(tsne_txt_pre[:, 0], tsne_txt_pre[:, 1], label='Text (Pre)', color='blue')
    axs[0].scatter(tsne_img_pre[:, 0], tsne_img_pre[:, 1], label='Image (Pre)', color='red')
    for i in range(len(texts)):
        axs[0].plot([tsne_txt_pre[i, 0], tsne_img_pre[i, 0]], [tsne_txt_pre[i, 1], tsne_img_pre[i, 1]], c='gray', alpha=0.4, linestyle='--')
    axs[0].legend()
    axs[0].set_title("Before Projector (PCA + t-SNE)")

    # after projector: 한 공간에서 매칭
    text_pts = tsne_post[:len(text_post)]
    img_pts = tsne_post[len(text_post):]
    axs[1].scatter(text_pts[:, 0], text_pts[:, 1], label='Text (Post)', color='blue')
    axs[1].scatter(img_pts[:, 0], img_pts[:, 1], label='Image (Post)', color='red')
    for i in range(len(text_post)):
        axs[1].plot([text_pts[i, 0], img_pts[i, 0]], [text_pts[i, 1], img_pts[i, 1]], c='gray', linestyle='--', alpha=0.4)
    axs[1].legend()
    axs[1].set_title("After Projector")
    axs[1].grid(True)


    # koclip
    txt_koclip_pts = tsne_koclip[:len(koclip_txt)]
    img_koclip_pts = tsne_koclip[len(koclip_txt):]

    axs[2].scatter(txt_koclip_pts[:, 0], txt_koclip_pts[:, 1], label='KoCLIP Text', color='blue')
    axs[2].scatter(img_koclip_pts[:, 0], img_koclip_pts[:, 1], label='KoCLIP Image', color='red')
    for i in range(len(koclip_txt)):
        axs[2].plot([txt_koclip_pts[i, 0], img_koclip_pts[i, 0]], [txt_koclip_pts[i, 1], img_koclip_pts[i, 1]], 
                    c='gray', linestyle='--', alpha=0.4)

    axs[2].legend()
    axs[2].set_title("KoCLIP")
    plt.savefig("/home/aikusrv01/C-MCR/visualization/before_hanclip_koclip_tsne.png")