"""Test BLIP-2 VQA answer extraction."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image

# Load an actual test image
img = Image.open(r"test_images/20250812-DSC02000.jpg").convert("RGB")
arr = np.array(img)
image_data = {"rgb_array": arr}

from imganalyzer.analysis.ai.local import LocalAI
result = LocalAI().analyze(image_data)
print("description:", result.get("description"))
print("scene_type:", result.get("scene_type"))
print("main_subject:", result.get("main_subject"))
print("lighting:", result.get("lighting"))
print("mood:", result.get("mood"))
print("keywords:", result.get("keywords"))
