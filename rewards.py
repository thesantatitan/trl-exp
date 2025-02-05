import cairosvg
import io
from PIL import Image
from typing import List, Tuple, Optional, Dict
import torch
import clip
import re

class SVGRewardFunction:
    def __init__(self,
                 format_weight: float = 1.0,
                 rendering_weight: float = 1.0,
                 clip_weight: float = 1.0,
                 device: Optional[str] = None):
        """Initialize the reward function with weights and CLIP model.

        Args:
            format_weight: Weight for format checking reward
            rendering_weight: Weight for rendering success reward
            clip_weight: Weight for CLIP similarity reward
            device: Device to run CLIP on ('cuda' or 'cpu')
        """
        self.format_weight = format_weight
        self.rendering_weight = rendering_weight
        self.clip_weight = clip_weight

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load CLIP model
        print(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def _format_check(self, completions: List[Dict]) -> Tuple[List[float], List[str]]:
        """Check format of completions and extract SVG content."""
        pattern = r'''
            ^
            <think>
            \s*
            (.*?)
            \s*
            </think>
            \s*
            (.*?)
            \s*
            (<svg>.*?</svg>)
            \s*
            (.*?)
            $
        '''

        completion_contents = [completion["content"] for completion in completions]
        results = []

        for content in completion_contents:
            match = re.match(pattern, content, re.VERBOSE)
            if match:
                results.append((1.0, match.group(3)))
            else:
                results.append((0.0, ""))

        scores, svg_contents = zip(*results)
        return list(scores), list(svg_contents)

    def _render_svg(self, svg_strings: List[str], format_scores: List[float]) -> Tuple[List[float], List[bytes]]:
        """Attempt to render SVGs to PNG."""
        results: List[Tuple[float, bytes]] = []

        for svg_string, format_score in zip(svg_strings, format_scores):
            if format_score == 0.0 or not svg_string:
                results.append((0.0, b''))
                continue

            try:
                png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
                if png_data is None:
                    results.append((0.0, b''))
                    continue

                Image.open(io.BytesIO(png_data))
                results.append((1.0, png_data))
            except Exception as e:
                print(f"Rendering failed: {e}")
                results.append((0.0, b''))

        rendering_scores, rendered_pngs = zip(*results)
        return list(rendering_scores), list(rendered_pngs)

    def _process_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to CLIP input tensor."""
        if not image_bytes:
            return torch.zeros(1, 512, device=self.device)

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features

        except Exception as e:
            print(f"Error processing image: {e}")
            return torch.zeros(1, 512, device=self.device)

    def _clip_similarity(self, rendered_pngs: List[bytes], ground_truth_embeddings: torch.Tensor) -> List[float]:
        """Calculate CLIP similarity scores."""
        ground_truth_embeddings = ground_truth_embeddings.to(self.device)
        similarity_scores = []

        for png_data in rendered_pngs:
            image_features = self._process_image(png_data)
            similarity = torch.nn.functional.cosine_similarity(
                image_features,
                ground_truth_embeddings
            ).item()
            similarity = max(0.0, min(1.0, similarity))
            similarity_scores.append(similarity)

        return similarity_scores

    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode ground truth images for comparison."""
        all_features = []

        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features)

        return torch.cat(all_features, dim=0)

    def __call__(self, completions: List[Dict], ground_truth_embeddings: torch.Tensor) -> List[float]:
        """Calculate combined reward scores for completions.

        Args:
            completions: List of completion dictionaries with 'content' key
            ground_truth_embeddings: Ground truth CLIP embeddings to compare against

        Returns:
            List of combined reward scores
        """
        # Check format and extract SVGs
        format_scores, svg_strings = self._format_check(completions)

        # Render SVGs to PNG
        rendering_scores, rendered_pngs = self._render_svg(svg_strings, format_scores)

        # Calculate CLIP similarity scores
        clip_scores = self._clip_similarity(rendered_pngs, ground_truth_embeddings)

        # Combine scores using weights
        final_scores = []
        for f, r, c in zip(format_scores, rendering_scores, clip_scores):
            weighted_score = (
                self.format_weight * f +
                self.rendering_weight * r +
                self.clip_weight * c
            ) / (self.format_weight + self.rendering_weight + self.clip_weight)
            final_scores.append(weighted_score)

        return final_scores
