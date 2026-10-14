"""
DermaAgent - Image Quality Assessment Agent
Evaluates skin lesion image quality: blurriness, illumination (dark/bright),
contrast, resolution, and readability. Suggests re-upload when quality is suboptimal.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try importing cv2 for fast Laplacian variance blur check; fallback to PIL/numpy if needed
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageQualityAgent:
    """
    Analyzes uploaded skin lesion images for clinical readability and quality factors:
    1. Blurriness (Laplacian variance)
    2. Underexposure (too dark)
    3. Overexposure (too bright / glare)
    4. Low contrast
    5. Minimum resolution
    """

    def __init__(self, blur_threshold=20.0, dark_threshold=40.0, bright_threshold=225.0, min_contrast=20.0, min_dim=120):
        self.name = "ImageQualityAgent"
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_contrast = min_contrast
        self.min_dim = min_dim

    def _compute_blur(self, gray_np):
        """Calculates sharpness score using Laplacian variance."""
        if HAS_CV2:
            return float(cv2.Laplacian(gray_np, cv2.CV_64F).var())
        # Pure numpy approximation of Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        pad = np.pad(gray_np.astype(np.float32), 1, mode='edge')
        lap = (
            pad[:-2, 1:-1] * kernel[0, 1] +
            pad[1:-1, :-2] * kernel[1, 0] +
            pad[1:-1, 1:-1] * kernel[1, 1] +
            pad[1:-1, 2:] * kernel[1, 2] +
            pad[2:, 1:-1] * kernel[2, 1]
        )
        return float(lap.var())

    def process(self, image_path):
        """
        Inspects image quality and returns structured quality report.
        """
        path = Path(image_path)
        if not path.exists():
            return {
                "agent": self.name,
                "is_acceptable": False,
                "quality_score": 0.0,
                "issues": ["Image file not found on disk."],
                "reupload_recommended": True,
                "advisory": "Image file could not be loaded. Please re-upload.",
                "metrics": {}
            }

        try:
            with Image.open(path) as pil_img:
                width, height = pil_img.size
                rgb_img = pil_img.convert("RGB")
                gray_img = rgb_img.convert("L")
                gray_np = np.array(gray_img)

            # 1. Resolution Check
            issues = []
            if width < self.min_dim or height < self.min_dim:
                issues.append(f"Image resolution is too low ({width}x{height}px; minimum {self.min_dim}x{self.min_dim}px recommended).")

            # 2. Brightness Check
            mean_brightness = float(gray_np.mean())
            if mean_brightness < self.dark_threshold:
                issues.append("Image is too dark (underexposed). Details of the lesion are shadowed.")
            elif mean_brightness > self.bright_threshold:
                issues.append("Image is overexposed (too bright / intense flash glare). Key lesion pigment details are washed out.")

            # 3. Contrast Check
            std_contrast = float(gray_np.std())
            if std_contrast < self.min_contrast:
                issues.append("Image has very low contrast. Lesion boundary is difficult to distinguish from background skin.")

            # 4. Blur / Sharpness Check
            sharpness = self._compute_blur(gray_np)
            if sharpness < self.blur_threshold:
                issues.append("Image appears blurry or out of focus. Fine clinical dermoscopic patterns are indistinct.")

            # Quality Score calculation (0 to 100)
            score = 100.0
            if sharpness < self.blur_threshold:
                score -= min(40.0, (self.blur_threshold - sharpness) * 0.8)
            if mean_brightness < self.dark_threshold:
                score -= min(30.0, (self.dark_threshold - mean_brightness) * 1.0)
            elif mean_brightness > self.bright_threshold:
                score -= min(30.0, (mean_brightness - self.bright_threshold) * 1.0)
            if std_contrast < self.min_contrast:
                score -= min(20.0, (self.min_contrast - std_contrast) * 1.0)
            if width < self.min_dim or height < self.min_dim:
                score -= 30.0

            quality_score = max(0.0, min(100.0, round(score, 1)))
            is_acceptable = (len(issues) == 0) or (quality_score >= 60.0 and sharpness >= (self.blur_threshold * 0.6))
            reupload_recommended = not is_acceptable

            if is_acceptable:
                if len(issues) == 0:
                    advisory = "Image clarity and illumination are optimal for screening."
                else:
                    advisory = f"Image quality is acceptable ({quality_score}/100), but could be improved for higher screening confidence."
            else:
                advisory = "Image quality is suboptimal for reliable screening. Please capture a clear, well-focused photo under good lighting without camera flash glare."

            return {
                "agent": self.name,
                "is_acceptable": is_acceptable,
                "quality_score": quality_score,
                "sharpness_level": "Sharp" if sharpness >= self.blur_threshold else "Blurry",
                "lighting_level": "Good" if (self.dark_threshold <= mean_brightness <= self.bright_threshold) else ("Too Dark" if mean_brightness < self.dark_threshold else "Too Bright"),
                "reupload_recommended": reupload_recommended,
                "issues": issues,
                "advisory": advisory,
                "metrics": {
                    "sharpness": round(sharpness, 1),
                    "mean_brightness": round(mean_brightness, 1),
                    "contrast": round(std_contrast, 1),
                    "dimensions": f"{width}x{height}"
                }
            }

        except Exception as e:
            return {
                "agent": self.name,
                "is_acceptable": False,
                "quality_score": 0.0,
                "issues": [f"Error processing image: {str(e)}"],
                "reupload_recommended": True,
                "advisory": "Could not verify image quality. Please upload a standard JPG or PNG photo.",
                "metrics": {}
            }


if __name__ == "__main__":
    agent = ImageQualityAgent()
    sample = PROJECT_ROOT / "dataset_real" / "test" / "akiec" / "ISIC_0024511.jpg"
    if sample.exists():
        res = agent.process(sample)
        print("[INFO] ImageQualityAgent Test:")
        print(f"Acceptable: {res['is_acceptable']}, Score: {res['quality_score']}, Sharpness: {res['metrics'].get('sharpness')}")
        print(f"Advisory: {res['advisory']}")
