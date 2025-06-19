import cairosvg
import io
from PIL import Image
from typing import List, Tuple, Optional, Dict
import torch
import clip
import re
from func_timeout import func_timeout, FunctionTimedOut
import numpy as np

class SVGRewardFunction:
    def __init__(self,
                 format_weight: float = 1.0,
                 rendering_weight: float = 1.0,
                 clip_weight: float = 1.0,
                 text_weight: float = 1.0,
                 mse_weight: float = 1.0,
                 device: Optional[str] = None):
        """Initialize the reward function with weights and CLIP model.

        Args:
            format_weight: Weight for format checking reward
            rendering_weight: Weight for rendering success reward
            clip_weight: Weight for CLIP similarity reward
            text_weight: Weight for text similarity reward
            mse_weight: Weight for MSE reward
            device: Device to run CLIP on ('cuda' or 'cpu')
        """
        self.format_weight = format_weight
        self.rendering_weight = rendering_weight
        self.clip_weight = clip_weight
        self.text_weight = text_weight
        self.mse_weight = mse_weight
        self.rewards = {"format": 0.0, "rendering": 0.0, "clip": 0.0, "text": 0.0, "mse": 0.0}
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
        # Strict pattern for format scoring
        strict_pattern = r'''
            ^
            <think>
            \s*
            (.*?)
            \s*
            </think>
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
        
        # Lenient pattern to extract SVG content even if format is not perfect
        lenient_svg_pattern = r'<generated_svg>\s*(.*?)\s*</generated_svg>'
        
        completion_contents = [completion[0]["content"] for completion in completions]
        results = []

        for content in completion_contents:
            # Check strict format for scoring
            strict_match = re.match(strict_pattern, content, re.VERBOSE | re.DOTALL)
            
            if strict_match:
                # Perfect format - full score and extract SVG
                results.append((1.0, strict_match.group(3)))
            else:
                # Try lenient extraction for SVG content
                lenient_match = re.search(lenient_svg_pattern, content, re.DOTALL)
                if lenient_match:
                    # Format is not perfect but SVG is present - partial score
                    results.append((0.5, lenient_match.group(1)))
                else:
                    # No SVG found at all - try to find any SVG-like content
                    svg_like = re.search(r'<svg[^>]*>.*?</svg>', content, re.DOTALL | re.IGNORECASE)
                    if svg_like:
                        results.append((0.25, svg_like.group(0)))
                    else:
                        results.append((0.0, ""))

        scores, svg_contents = zip(*results)
        return list(scores), list(svg_contents)

    def _render_svg(self, svg_strings: List[str], format_scores: List[float]) -> Tuple[List[float], List[bytes]]:
        """Attempt to render SVGs to PNG."""
        results: List[Tuple[float, bytes]] = []

        for svg_string, format_score in zip(svg_strings, format_scores):
            # Remove the format score dependency - try to render any SVG content
            if not svg_string:
                results.append((0.0, None))
                continue

            try:
                png_data = func_timeout(10, cairosvg.svg2png, args=(svg_string.encode('utf-8'),))
                if png_data is None:
                    results.append((0.0, b''))
                    continue

                Image.open(io.BytesIO(png_data))
                results.append((1.0, png_data))
            except FunctionTimedOut:
                print("SVG rendering timed out")
                results.append((0.0, b''))
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
            if not png_data:
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
            if not png_data:
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

    def _calculate_mse(self, rendered_pngs: List[bytes], input_images: List[Image.Image]) -> List[float]:
        """Calculate MSE between rendered SVGs and input images, scaled to 0-1.
        
        Args:
            rendered_pngs: List of rendered PNG bytes
            input_images: List of PIL Image objects to compare against
            
        Returns:
            List of MSE scores (0-1, where 1 is best/lowest error)
        """
        mse_scores = []
        
        for png_data, input_img in zip(rendered_pngs, input_images):
            if not png_data:
                mse_scores.append(0.0)
                continue
                
            try:
                # Convert rendered PNG to PIL Image
                rendered_img = Image.open(io.BytesIO(png_data)).convert('RGB')
                
                # Resize input image to match rendered image size
                input_img_resized = input_img.convert('RGB').resize(rendered_img.size, Image.Resampling.LANCZOS)
                
                # Convert to numpy arrays
                rendered_arr = np.array(rendered_img, dtype=np.float32) / 255.0
                input_arr = np.array(input_img_resized, dtype=np.float32) / 255.0
                
                # Calculate MSE
                mse = np.mean((rendered_arr - input_arr) ** 2)
                
                # Scale MSE to 0-1 range (1 is best, 0 is worst)
                # Using exponential decay: score = exp(-k * mse)
                # k=10 means MSE of 0.1 gives score ~0.37
                mse_score = np.exp(-10 * mse)
                
                mse_scores.append(float(mse_score))
                
            except Exception as e:
                print(f"Error calculating MSE: {e}")
                mse_scores.append(0.0)
                
        return mse_scores

    def __call__(self, prompts: List[Dict], completions: List[Dict], ground_truth_embeddings: List[torch.Tensor], text_embeddings: Optional[List[torch.Tensor]] = None, input_images: Optional[List[Image.Image]] = None, mse_reward: bool = False) -> List[float]:
        """Calculate combined reward scores for completions.

        Args:
            prompts: List of prompt dictionaries
            completions: List of completion dictionaries with 'content' key
            ground_truth_embeddings: Ground truth CLIP embeddings to compare against (Tensor or List)
            text_embeddings: Optional text embeddings to compare against (Tensor or List)
            input_images: Optional list of input images for MSE calculation
            mse_reward: Whether to calculate MSE reward (requires input_images)

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

        # Handle MSE reward
        if mse_reward:
            if input_images is None:
                raise ValueError("MSE reward requested but no input images provided")
            mse_scores = self._calculate_mse(rendered_pngs, input_images)
        else:
            mse_scores = [0.0] * len(completions)
        self.rewards["mse"] += sum(mse_scores)

        # Combine scores using weights
        final_scores = []
        total_weight = self.format_weight + self.rendering_weight + self.clip_weight + self.text_weight
        if mse_reward:
            total_weight += self.mse_weight
            
        for f, r, c, t, m in zip(format_scores, rendering_scores, clip_scores, text_scores, mse_scores):
            weighted_score = (
                self.format_weight * f +
                self.rendering_weight * r +
                self.clip_weight * c +
                self.text_weight * t +
                (self.mse_weight * m if mse_reward else 0)
            ) / total_weight
            final_scores.append(weighted_score)
        # print(self.rewards)
        # print(final_scores)
        return final_scores
