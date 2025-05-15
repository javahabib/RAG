# app/main.py
import gradio as gr
from app.load_models import load_text_model, load_clip_model, load_faiss_indices
from app.load_models import load_embeddings
from app.retriever import search_text, search_image
from app.generator import generate_answer
from PIL import Image
import os
import pickle
import numpy as np

# === Load everything ===
text_model = load_text_model()
clip_model, clip_processor = load_clip_model()

text_index, image_index = load_faiss_indices(
    "data/text_index.faiss", "data/image_index.faiss"
)



text_embeddings, image_embeddings = load_embeddings(
    "data/text_embeddings.npy", "data/image_embeddings.npy"
)

# Load original text chunks from file
with open("data/text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

def query_system(text_input=None, image_input=None):
    if text_input and not image_input:
        # Search text
        top_indices = search_text(text_input, text_model, text_index)
        context_chunks = [text_chunks[i] for i in top_indices]
        answer = generate_answer(context_chunks, text_input)
        return answer
    
    elif image_input and not text_input:
        # Search image: directly use the image embeddings
        top_indices = search_image(image_input, clip_model, clip_processor, image_index)
        context_images = [image_embeddings[i] for i in top_indices]  # retrieve embeddings
        prompt = "Describe the content of this image."  # or another prompt for context
        answer = generate_answer(context_images, prompt)
        return answer
    
    else:
        return "Please provide either text or image input, not both."



# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("# **Multimodal RAG System**")
    gr.Markdown("Enter a query or upload an image to get answers.")

    with gr.Tabs("Text Query"):
        text_input = gr.Textbox(label="Enter Your Query")
        text_output = gr.Textbox(label="Answer", interactive=False)
        text_submit = gr.Button("Submit Text Query")
        text_submit.click(query_system, inputs=text_input, outputs=text_output)

    with gr.Tabs("Image Query"):
        image_input = gr.Image(label="Upload an Image", type="pil")
        image_output = gr.Textbox(label="Answer", interactive=False)
        image_submit = gr.Button("Submit Image Query")
        image_submit.click(query_system, inputs=image_input, outputs=image_output)

# Run the Gradio app
demo.launch()


#to run code : python -m app.main