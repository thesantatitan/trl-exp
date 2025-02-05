import torch
from PIL import Image
import clip
import sys

def truncate_text_to_fit_context(text, max_words=50):
    # Simple word-based truncation
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

def compute_text_image_similarity(text, image_path):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Truncate text before tokenization
    truncated_text = truncate_text_to_fit_context(text)

    try:
        # Tokenize the truncated text
        tokenized = clip.tokenize([truncated_text]).to(device)

        # Load and preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing inputs: {e}")
        return None

    # Get features
    with torch.no_grad():
        text_features = model.encode_text(tokenized)
        image_features = model.encode_image(image)

        # Normalize the features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity (cosine similarity)
        similarity = (text_features @ image_features.T).item()

    return similarity, truncated_text

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <text> <image_path>")
        sys.exit(1)

    text = sys.argv[1]
    image_path = sys.argv[2]

    result = compute_text_image_similarity(text, image_path)
    if result is not None:
        similarity, truncated_text = result
        print(f"Truncated text used: {truncated_text}")
        print(f"Similarity score between text and image: {similarity:.4f}")
