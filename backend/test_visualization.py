
from pathlib import Path

import numpy as np
from PIL import Image

from app.inference import (
    ChestXrayInference,
    CLASS_NAMES,
)
from app.preprocessing import preprocess_dicom


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "train"
    / "vindr_300"
)

OUTPUT_DIR = BACKEND_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. VISUALIZATION COLORS
# --------------------------------------------------

CLASS_COLORS = {
    "Aortic enlargement": (255, 60, 60),     # Red
    "Cardiomegaly": (50, 220, 80),           # Green
    "Pleural thickening": (50, 150, 255),    # Blue
    "Pulmonary fibrosis": (255, 210, 40),    # Yellow
    "Lung Opacity": (220, 60, 220),          # Magenta
}

ALPHA = 0.45


# --------------------------------------------------
# 3. OVERLAY FUNCTION
# --------------------------------------------------

def create_overlay(background, mask, color, alpha=ALPHA):

    background = background.astype(np.float32)
    overlay = background.copy()

    color_array = np.array(color, dtype=np.float32)

    # Apply color only where mask is positive
    overlay[mask > 0] = (
        (1 - alpha) * background[mask > 0]
        + alpha * color_array
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


# --------------------------------------------------
# 4. FIND TEST DICOM
# --------------------------------------------------

dicom_files = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in [".dcm", ".dicom"]
])

if not dicom_files:
    raise FileNotFoundError(
        f"No DICOM files found in {IMAGE_DIR}"
    )

dicom_path = dicom_files[0]

print("Testing DICOM:", dicom_path.name)


# --------------------------------------------------
# 5. LOAD MODEL AND RUN INFERENCE
# --------------------------------------------------

inference = ChestXrayInference()

result = inference.predict(dicom_path)

print("\nInference completed!")


# --------------------------------------------------
# 6. GET PREPROCESSED X-RAY
# --------------------------------------------------

image_tensor, original_size, ds = preprocess_dicom(
    dicom_path
)

# Convert [1, 1, H, W] into a grayscale image
image_array = image_tensor[0, 0].numpy()

image_uint8 = (
    np.clip(image_array, 0, 1) * 255
).round().astype(np.uint8)

# Convert grayscale to RGB
background = np.stack(
    [image_uint8] * 3,
    axis=-1
)

# Save preprocessed X-ray
Image.fromarray(image_uint8).save(
    OUTPUT_DIR / "original_xray.png"
)


# --------------------------------------------------
# 7. GENERATE INDIVIDUAL SEGMENTATION OVERLAYS
# --------------------------------------------------

combined_overlay = background.copy()

for class_name in CLASS_NAMES:

    mask = result["masks"][class_name]

    color = CLASS_COLORS[class_name]

    # Individual overlay
    individual_overlay = create_overlay(
        background,
        mask,
        color
    )

    safe_name = (
        class_name.lower()
        .replace(" ", "_")
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_overlay.png"
    )

    Image.fromarray(individual_overlay).save(
        output_path
    )

    # Add this class to combined overlay
    combined_overlay = create_overlay(
        combined_overlay,
        mask,
        color
    )

    print(
        f"{class_name}: "
        f"{int(mask.sum())} positive pixels"
    )


# --------------------------------------------------
# 8. SAVE COMBINED OVERLAY
# --------------------------------------------------

Image.fromarray(combined_overlay).save(
    OUTPUT_DIR / "combined_overlay.png"
)


# --------------------------------------------------
# 9. SAVE PROBABILITY MAPS
# --------------------------------------------------

for index, class_name in enumerate(CLASS_NAMES):

    probability_map = result["probabilities"][index]

    probability_image = (
        np.clip(probability_map, 0, 1) * 255
    ).round().astype(np.uint8)

    safe_name = (
        class_name.lower()
        .replace(" ", "_")
    )

    Image.fromarray(probability_image).save(
        OUTPUT_DIR / f"{safe_name}_probability.png"
    )


# --------------------------------------------------
# 10. SUMMARY
# --------------------------------------------------

print("\nVISUALIZATION COMPLETED!")
print("------------------------")
print("Original DICOM dimensions:", original_size)
print("Model input dimensions:", image_array.shape)
print("Output directory:", OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print("-", file.name)

print("\nVISUALIZATION TEST PASSED!")
