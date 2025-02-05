import cairosvg
import io
from PIL import Image
from typing import List, Tuple, Optional
import torch
import clip
import re

def format_reward_func(completion_contents, **kwargs):
    pattern = r'''
        ^                   # Start of string
        <think>         # Literal <thinking>
        \s*                # Optional whitespace
        (.*?)              # Any text (non-greedy) for thinking content
        \s*                # Optional whitespace
        </think>        # Literal </thinking>
        \s*                # Optional whitespace
        (.*?)              # Any text between thinking and svg
        \s*                # Optional whitespace
        (<svg>.*?</svg>)   # Complete SVG tag and content
        \s*                # Optional whitespace
        (.*?)              # Any remaining text
        $                  # End of string
    '''
    results = []
    for content in completion_contents:
        match = re.match(pattern, content, re.VERBOSE)
        if match:
            results.append((1.0, match.group(3)))  # group(3) now contains the complete svg tag
        else:
            results.append((0.0, ""))

    scores, svg_contents = zip(*results)
    return scores, svg_contents



def rendering_reward_func(svg_strings: List[str], format_scores: List[float]) -> Tuple[List[float], List[bytes]]:
    results: List[Tuple[float, bytes]] = []

    for svg_string, format_score in zip(svg_strings, format_scores):
        # If format score is 0 or svg string is empty, return 0 score and empty bytes
        if format_score == 0.0 or not svg_string:
            results.append((0.0, b''))
            continue

        try:
            # Try to render the SVG to PNG using cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
            if png_data is None:
                results.append((0.0, b''))
                continue

            # Convert to PIL Image to verify it's a valid image
            Image.open(io.BytesIO(png_data))
            results.append((1.0, png_data))
        except Exception as e:
            # If rendering fails, return 0 score and empty bytes
            print(f"Rendering failed: {e}")
            results.append((0.0, b''))

    # Unzip the results into separate lists
    rendering_scores, rendered_svgs = zip(*results)
    return list(rendering_scores), list(rendered_svgs)

class ClipRewardFunc:
    def __init__(self, device: Optional[str] = None):
        """Initialize CLIP model and preprocessing transform.

        Args:
            device: Device to run CLIP on ('cuda' or 'cpu'). If None, will use CUDA if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load CLIP model and move to specified device
        print(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Put model in eval mode
        self.model.eval()

    def _process_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to CLIP input tensor."""
        if not image_bytes:
            return torch.zeros(1, 512, device=self.device)

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Get image features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features

        except Exception as e:
            print(f"Error processing image: {e}")
            return torch.zeros(1, 512, device=self.device)

    def __call__(self,
                 rendered_svgs: List[bytes],
                 ground_truth_clip_embeddings: torch.Tensor) -> List[float]:
        """Calculate CLIP similarity scores between rendered SVGs and ground truth embeddings.

        Args:
            rendered_svgs: List of PNG image bytes from the rendering function
            ground_truth_clip_embeddings: Ground truth CLIP embeddings to compare against
                                        (should be normalized, shape: [1, 512])

        Returns:
            List of cosine similarity scores between 0 and 1
        """
        # Ensure ground truth embeddings are on the correct device
        ground_truth_clip_embeddings = ground_truth_clip_embeddings.to(self.device)

        similarity_scores = []

        # Process each rendered SVG
        for svg_png in rendered_svgs:
            # Get image features
            image_features = self._process_image(svg_png)

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                image_features,
                ground_truth_clip_embeddings
            ).item()

            # Clip similarity to [0, 1] range and append to results
            similarity = max(0.0, min(1.0, similarity))
            similarity_scores.append(similarity)

        return similarity_scores

    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Utility method to encode ground truth images for later comparison.

        Args:
            image_paths: List of paths to ground truth images

        Returns:
            Tensor of normalized CLIP embeddings
        """
        all_features = []

        for image_path in image_paths:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Get features
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features)

        return torch.cat(all_features, dim=0)
