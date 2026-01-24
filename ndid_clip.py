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


def find_duplicates(query_folder, embeddings_db, threshold=0.93):
    predicted_duplicates = {}

    for img_name in os.listdir(query_folder):
        img_path = os.path.join(query_folder, img_name)

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

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

        # ✅ THIS IS WHERE YOUR CODE GOES
        if best_score >= threshold:
            q = os.path.basename(img_path)
            m = os.path.basename(best_match)

            predicted_duplicates.setdefault(q, set()).add(m)

    return predicted_duplicates



def remove_duplicate_images(image_folder, threshold=0.95):
    print("\n🔹 Running Storage Optimization (Deleting Duplicates)...\n")

    # Step 1: Build embeddings for all images
    embeddings = {}
    image_paths = []

    for img in os.listdir(image_folder):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_folder, img)
            emb = get_embedding(path)
            if emb is not None:
                embeddings[path] = emb
                image_paths.append(path)

    # Step 2: Find duplicates
    to_delete = set()

    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            img1 = image_paths[i]
            img2 = image_paths[j]

            sim = cosine_similarity(
                [embeddings[img1]],
                [embeddings[img2]]
            )[0][0]

            if sim >= threshold:
                # Mark second image as duplicate
                to_delete.add(img2)

    # Step 3: Delete duplicates
    print("\nDeleting duplicates:\n")
    for path in to_delete:
        print(f"🗑️ Deleting: {path}")
        os.remove(path)

    print(f"\n✅ Storage Optimization Done. Deleted {len(to_delete)} images.\n")
    return to_delete

def search_different(query_image, embeddings_db, top_k=5, diversity_threshold=0.85):
    query_emb = get_embedding(query_image)
    if query_emb is None:
        return []

    scores = []

    # Step 1: Compute similarity with all images
    for stored_path, stored_emb in embeddings_db.items():
        sim = cosine_similarity([query_emb], [stored_emb])[0][0]
        if(sim==1):
            continue
        scores.append((stored_path, sim))

    # Step 2: Sort (low → high similarity)
    scores = sorted(scores, key=lambda x: x[1],reverse=True)

    # Step 3: Pick top results but remove near-duplicates among them
    selected = []

    for path, sim in scores[:top_k]:   # take highest similarity ones
        keep = True
        sim_between=sim
        for sel_path, _ in selected:
            sim_between = cosine_similarity(
                [embeddings_db[path]],
                [embeddings_db[sel_path]]
            )[0][0]

            # If two retrieved images are too similar → skip one
            if sim_between >= diversity_threshold:
                keep = False
                break

        if keep:
            selected.append((path, sim))
    if(len(selected)==0):
        print("No similar but differnt image found..")
    print("\n🔍 Different (Diverse) Images Retrieved:")
    for i, (path, sim) in enumerate(selected):
        print(f"{i+1}. {path} → similarity={sim:.3f}")

    return selected

def store_if_unique(image_path, embeddings_db, threshold=0.95):
    new_emb = get_embedding(image_path)
    if new_emb is None:
        return False

    # If DB empty, store directly
    if len(embeddings_db) == 0:
        embeddings_db[image_path] = new_emb
        print(f"Stored (first image): {image_path}")
        return True

    for stored_path, stored_emb in embeddings_db.items():
        sim = cosine_similarity([new_emb], [stored_emb])[0][0]

        if sim >= threshold:
            print(f"⚠️ REPOST DETECTED! ❌ NOT STORED (duplicate of {stored_path}), sim={sim:.3f}")
            return False

    embeddings_db[image_path] = new_emb
    print(f"✅ STORED: {image_path}")
    return True

def compute_f1(predicted, ground_truth):
    TP = 0
    FP = 0
    FN = 0

    # Evaluate only on images present in ground truth
    for img in ground_truth:
        gt_set = ground_truth.get(img, set())
        pred_set = predicted.get(img, set())

        TP += len(gt_set & pred_set)   # correct matches
        FP += len(pred_set - gt_set)   # wrong matches
        FN += len(gt_set - pred_set)   # missed matches

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return precision, recall, f1


landmarks_folder = "datasets/landmarks"

# landmarks_db = build_database(landmarks_folder)
   
# find_duplicates(landmarks_folder, landmarks_db, threshold=0.95)

original_folder = "datasets/copydays/original"
transformed_folder = "datasets/copydays/transformed"

copydays_db = build_database(original_folder)
trans_folder=build_database(transformed_folder)
# print(f"\nTotal near-duplicates found: {len(results)}")
# remove_duplicate_images("datasets/copydays/original", threshold=0.95)

search_different("datasets/landmarks/test.jpg",copydays_db, top_k=3)

store_if_unique("datasets/landmarks/test.jpg", copydays_db)

ground_truth = {
    "img1.jpg": {
        "img1.jpg"
    },
    "img2.jpg": {
        "img2.jpg"
    }
}

results = find_duplicates("datasets/copydays/test",copydays_db, threshold=0.75)

p,r,f1=compute_f1(results,ground_truth)


print("F1:",f1)

