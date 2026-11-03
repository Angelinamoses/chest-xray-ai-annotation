
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image


def preprocess_dicom(dicom_path, image_size=256):
    """
    Preprocess a VinDr-CXR DICOM image using the same
    procedure used during model training.

    Returns:
        image_tensor: Tensor [1, 1, 256, 256]
        original_size: Tuple (height, width)
        ds: Original DICOM dataset
    """

    dicom_path = Path(dicom_path)

    # 1. Read DICOM
    ds = pydicom.dcmread(str(dicom_path))

    # 2. Extract pixel array as float32
    image = ds.pixel_array.astype(np.float32)

    # 3. Ensure 2D image
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D image, got shape {image.shape}: "
            f"{dicom_path.name}"
        )

    # Preserve original dimensions
    original_size = image.shape

    # 4. Apply modality rescaling
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    image = image * slope + intercept

    # 5. Handle MONOCHROME1
    photometric = getattr(ds, "PhotometricInterpretation", "")

    if photometric == "MONOCHROME1":
        image = image.max() - image

    # 6. Robust intensity normalization
    lower = np.percentile(image, 1)
    upper = np.percentile(image, 99)

    if upper <= lower:
        raise ValueError(
            f"Invalid intensity range: {dicom_path.name}"
        )

    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)

    # 7. Resize to 256 x 256
    pil_image = Image.fromarray(image.astype(np.float32))

    pil_image = pil_image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR
    )

    image = np.asarray(pil_image, dtype=np.float32)

    # 8. Explicitly create a writable, contiguous copy
    image = np.array(
        image,
        dtype=np.float32,
        copy=True,
        order="C"
    )

    # 9. Convert to PyTorch tensor
    image_tensor = torch.from_numpy(image)

    # [H, W] -> [1, 1, H, W]
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    return image_tensor, original_size, ds
