import os

# Define paths relative to your project root
primary_path = "../models/trained_models/crop_weed_model.pt"
fallback_path = "../training/runs/detect/desktop_crop_weed2/weights/best.pt"

print("Checking model paths...\n")

print("Primary path (crop_weed_model.pt):")
print("  Absolute:", os.path.abspath(primary_path))
print("  Exists?:", os.path.exists(primary_path))

print("\nFallback path (best.pt):")
print("  Absolute:", os.path.abspath(fallback_path))
print("  Exists?:", os.path.exists(fallback_path))
