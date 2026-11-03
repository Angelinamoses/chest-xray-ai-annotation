
from pathlib import Path
import torch

from app.preprocessing import preprocess_dicom

# Locate the dataset directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "train" / "vindr_300"

# Find the first DICOM file
dicom_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in [".dcm", ".dicom"]
]

if not dicom_files:
    raise FileNotFoundError(f"No DICOM files found in {IMAGE_DIR}")

dicom_path = dicom_files[0]

print("Testing file:", dicom_path.name)

# Run preprocessing
image_tensor, original_size, ds = preprocess_dicom(dicom_path)

print("\nPREPROCESSING RESULTS")
print("---------------------")
print("Tensor shape:", image_tensor.shape)
print("Tensor dtype:", image_tensor.dtype)
print("Original image size:", original_size)
print("Photometric interpretation:",
      getattr(ds, "PhotometricInterpretation", "Unknown"))
print("Minimum intensity:", image_tensor.min().item())
print("Maximum intensity:", image_tensor.max().item())
print("All values finite:", torch.isfinite(image_tensor).all().item())

# Assertions
assert image_tensor.shape == (1, 1, 256, 256)
assert image_tensor.dtype == torch.float32
assert torch.isfinite(image_tensor).all()
assert image_tensor.min() >= 0
assert image_tensor.max() <= 1

print("\nPREPROCESSING TEST PASSED!")
