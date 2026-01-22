/////////FOR MID EVAL.////

import os
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_embedding(image_path):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def build_database(image_folder):
    embeddings_db = {}
    
    print(f"\nBuilding database from: {image_folder}\n")

    for img_name in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)

        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            emb = get_embedding(img_path)
            if emb is not None:
                embeddings_db[img_path] = emb

    print(f"\nTotal images stored: {len(embeddings_db)}\n")
    return embeddings_db


def find_duplicates(query_folder, embeddings_db, threshold=0.9):
    print("\nChecking for duplicates...\n")

    results = []

    for img_name in tqdm(os.listdir(query_folder)):
        img_path = os.path.join(query_folder, img_name)

        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            new_emb = get_embedding(img_path)
            if new_emb is None:
                continue

            best_match = None
            best_score = 0

            for stored_path, stored_emb in embeddings_db.items():
                sim = cosine_similarity([new_emb], [stored_emb])[0][0]

                if sim > best_score:
                    best_score = sim
                    best_match = stored_path

            if best_score >= threshold:
                print(f"DUPLICATE: {img_path}")
                print(f"Matched with: {best_match}")
                print(f"Similarity: {best_score:.4f}\n")

                results.append((img_path, best_match, best_score))

    return results
  
landmarks_folder = "datasets/landmarks"

landmarks_db = build_database(landmarks_folder)

   
find_duplicates(landmarks_folder, landmarks_db, threshold=0.95)


original_folder = "datasets/copydays/original"
transformed_folder = "datasets/copydays/transformed"

copydays_db = build_database(original_folder)

results = find_duplicates(transformed_folder, copydays_db, threshold=0.75)


print(f"\nTotal near-duplicates found: {len(results)}")
