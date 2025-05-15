# app/retriever.py
import numpy as np

def search_text(query, model, faiss_index, top_k=5):
    query_vec = model.encode([query])
    D, I = faiss_index.search(np.array(query_vec), top_k)
    return I[0]  # return top indices

def search_image(image_pil, clip_model, clip_processor, faiss_index, top_k=5):
    inputs = clip_processor(images=image_pil, return_tensors="pt")
    features = clip_model.get_image_features(**inputs)
    image_vec = features.detach().numpy()
    D, I = faiss_index.search(image_vec, top_k)
    return I[0]
