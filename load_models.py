# app/load_models.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

def load_text_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_faiss_indices(text_index_path, image_index_path):
    text_index = faiss.read_index(text_index_path)
    image_index = faiss.read_index(image_index_path)
    return text_index, image_index

def load_embeddings(text_emb_path, image_emb_path):
    text_embeddings = np.load(text_emb_path)
    image_embeddings = np.load(image_emb_path)
    return text_embeddings, image_embeddings
