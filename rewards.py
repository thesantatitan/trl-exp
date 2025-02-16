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
                 text_weight: float = 1.0,
                 device: Optional[str] = None):
        """Initialize the reward function with weights and CLIP model.

        Args:
            format_weight: Weight for format checking reward
            rendering_weight: Weight for rendering success reward
            clip_weight: Weight for CLIP similarity reward
            text_weight: Weight for text similarity reward
            device: Device to run CLIP on ('cuda' or 'cpu')
        """
        self.format_weight = format_weight
        self.rendering_weight = rendering_weight
        self.clip_weight = clip_weight
        self.text_weight = text_weight
        self.rewards = {"format": 0.0, "rendering": 0.0, "clip": 0.0, "text": 0.0}
        self.count_since_logged = 0
        self.__name__ = "SVGRewardFunction"

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
            <reasoning>
            \s*
            (.*?)
            \s*
            </reasoning>
            \s*
            (.*?)
            \s*
            <generated_svg>
            \s*
            (.*?)
            \s*
            </generated_svg>
            \s*
            (.*?)
            $
        '''
        # print(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
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

    def _clip_similarity(self, rendered_pngs: List[bytes], ground_truth_embeddings: List[torch.Tensor]) -> List[float]:
        """Calculate CLIP similarity scores."""
        ground_truth_embeddings = [torch.tensor(g, device=self.device) for g in ground_truth_embeddings]
        similarity_scores = []

        for i,png_data in enumerate(rendered_pngs):
            if png_data == b'':
                similarity_scores.append(0.0)
            else:
                image_features = self._process_image(png_data)
                similarity = torch.nn.functional.cosine_similarity(
                    image_features,
                    ground_truth_embeddings[i]
                ).item()
                similarity = max(0.0, min(1.0, similarity))
                similarity_scores.append(similarity)

        return similarity_scores

    def _clip_text_reward_func(self, rendered_pngs: List[bytes], text_embeddings: List[torch.Tensor]) -> List[float]:
        """Calculate CLIP similarity scores between rendered images and text embeddings."""
        text_embeddings = [torch.tensor(t, device=self.device) for t in text_embeddings]
        similarity_scores = []

        for i,png_data in enumerate(rendered_pngs):
            if png_data == b'':
                similarity_scores.append(0.0)
            else:
                image_features = self._process_image(png_data)
                similarity = torch.nn.functional.cosine_similarity(
                    image_features,
                    text_embeddings[i]
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

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts for comparison."""
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def __call__(self, prompts: List[Dict], completions: List[Dict], ground_truth_embeddings: List[torch.Tensor], text_embeddings: Optional[List[torch.Tensor]] = None) -> List[float]:
        """Calculate combined reward scores for completions.

        Args:
            completions: List of completion dictionaries with 'content' key
            ground_truth_embeddings: Ground truth CLIP embeddings to compare against (Tensor or List)
            text_embeddings: Optional text embeddings to compare against (Tensor or List)

        Returns:
            List of combined reward scores
        """
        self.count_since_logged += len(completions)

        # Check format and extract SVGs
        format_scores, svg_strings = self._format_check(completions)
        self.rewards["format"] += sum(format_scores)
        # Render SVGs to PNG
        rendering_scores, rendered_pngs = self._render_svg(svg_strings, format_scores)
        self.rewards["rendering"] += sum(rendering_scores)

        # Calculate CLIP similarity scores
        clip_scores = self._clip_similarity(rendered_pngs, ground_truth_embeddings)
        self.rewards["clip"] += sum(clip_scores)

        # Handle text embeddings
        if text_embeddings is not None:
            text_scores = self._clip_text_reward_func(rendered_pngs, text_embeddings)
        else:
            text_scores = [0.0] * len(completions)
        self.rewards["text"] += sum(text_scores)

        # Combine scores using weights
        final_scores = []
        for f, r, c, t in zip(format_scores, rendering_scores, clip_scores, text_scores):
            weighted_score = (
                self.format_weight * f +
                self.rendering_weight * r +
                self.clip_weight * c +
                self.text_weight * t
            ) / (self.format_weight + self.rendering_weight + self.clip_weight + self.text_weight)
            final_scores.append(weighted_score)

        return final_scores
