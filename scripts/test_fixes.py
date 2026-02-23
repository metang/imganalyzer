"""Quick test of aesthetic scorer and object detection fixes."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

# Create a simple test image (landscape-ish gradient)
arr = np.zeros((480, 640, 3), dtype=np.uint8)
for i in range(480):
    arr[i, :, 2] = int(255 * i / 479)  # blue gradient (sky-like)
    arr[i, :, 1] = int(100 * (1 - i / 479))
arr[:, :, 0] = 30  # slight red tint

image_data = {"rgb_array": arr}

print("=" * 60)
print("TEST 1: Aesthetic Scorer")
print("=" * 60)
try:
    from imganalyzer.analysis.ai.aesthetic import AestheticScorer
    scorer = AestheticScorer()
    result = scorer.analyze(image_data)
    print(f"  aesthetic_score: {result['aesthetic_score']}")
    print(f"  aesthetic_label: {result['aesthetic_label']}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 2: Object Detector (dtype fix)")
print("=" * 60)
try:
    # Use a photo-like image with a "person" (blue rectangle as placeholder)
    arr2 = arr.copy()
    # Draw a tall rectangle to simulate a person silhouette
    arr2[100:400, 280:360, :] = [200, 150, 100]  # brownish person shape

    image_data2 = {"rgb_array": arr2}
    from imganalyzer.analysis.ai.objects import ObjectDetector
    detector = ObjectDetector()
    result2 = detector.analyze(image_data2, prompt="person . sky . gradient .")
    print(f"  detected_objects: {result2['detected_objects']}")
    print(f"  has_person: {result2['has_person']}")
    print("  PASS (no dtype warning expected)")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()

print()
print("All tests complete.")
