import torch
from PIL import Image
import clip
import sys

def compute_image_similarity(image_path1, image_path2):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load and preprocess the images
    try:
        image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading images: {e}")
        return None

    # Get image features
    with torch.no_grad():
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)

    # Normalize the features
    image1_features /= image1_features.norm(dim=-1, keepdim=True)
    image2_features /= image2_features.norm(dim=-1, keepdim=True)

    # Compute similarity (cosine similarity)
    similarity = (image1_features @ image2_features.T).item()

    return similarity

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_path1> <image_path2>")
        sys.exit(1)

    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]

    similarity = compute_image_similarity(image_path1, image_path2)
    if similarity is not None:
        print(f"Similarity score between images: {similarity:.4f}")
        # The similarity score ranges from -1 to 1, where:
        # 1 means very similar
        # -1 means very different
        # 0 means neutral
