# app/generator.py

import requests
import numpy as np

OLLAMA_URL = "your url "
MODEL_NAME = "phi"

def generate_answer(context, query):
    # If the context is image embeddings, we will not join it as text
    # We need to describe the images or summarize them based on the embeddings
    if isinstance(context, list) and isinstance(context[0], np.ndarray):
        # Process image embeddings to a description
        context_description = " ".join([f"Image {i+1}: Similar content based on embeddings" for i in range(len(context))])
    else:
        # Regular case for text-based context
        context_description = " ".join(context)

    # Construct the prompt for the model
    prompt = f"Answer the following question based on the context:\n\nContext: {context_description}\n\nQuestion: {query}"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    result = response.json()

    return result["response"]
