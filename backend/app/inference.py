
from pathlib import Path

import numpy as np
import torch

from app.model import UNet
from app.preprocessing import preprocess_dicom


# --------------------------------------------------
# 1. PATHS AND MODEL CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    BACKEND_DIR
    / "weights"
    / "best_unet_baseline.pth"
)

CLASS_NAMES = [
    "Aortic enlargement",
    "Cardiomegaly",
    "Pleural thickening",
    "Pulmonary fibrosis",
    "Lung Opacity",
]

THRESHOLDS = {
    "Aortic enlargement": 0.60,
    "Cardiomegaly": 0.70,
    "Pleural thickening": 0.10,
    "Pulmonary fibrosis": 0.10,
    "Lung Opacity": 0.15,
}


# --------------------------------------------------
# 2. INFERENCE ENGINE
# --------------------------------------------------

class ChestXrayInference:

    def __init__(self):

        # Select available device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Check checkpoint
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        # Initialize exact trained architecture
        self.model = UNet(
            in_channels=1,
            num_classes=5
        ).to(self.device)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True
        )

        # Extract model weights from the training checkpoint
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Evaluation mode
        self.model.eval()

        print("Model loaded successfully!")
        print("Device:", self.device)
        print("Trainable parameters:", sum(
            p.numel() for p in self.model.parameters()
        ))

    # --------------------------------------------------
    # 3. PREDICTION
    # --------------------------------------------------

    @torch.inference_mode()
    def predict(self, dicom_path):

        # Preprocess DICOM
        image_tensor, original_size, ds = preprocess_dicom(
            dicom_path
        )

        image_tensor = image_tensor.to(self.device)

        # Forward pass
        logits = self.model(image_tensor)

        # Convert logits into probabilities
        probabilities = torch.sigmoid(logits)

        # Move predictions to CPU
        probability_array = (
            probabilities[0].cpu().numpy()
        )

        # Generate binary segmentation masks
        masks = {}

        for index, class_name in enumerate(CLASS_NAMES):

            threshold = THRESHOLDS[class_name]

            binary_mask = (
                probability_array[index] >= threshold
            ).astype(np.uint8)

            masks[class_name] = binary_mask

        return {
            "masks": masks,
            "probabilities": probability_array,
            "original_size": original_size,
            "input_size": (256, 256),
            "class_names": CLASS_NAMES,
            "thresholds": THRESHOLDS,
            "device": str(self.device),
        }


# --------------------------------------------------
# 4. STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = BACKEND_DIR.parent

    IMAGE_DIR = (
        PROJECT_ROOT
        / "data"
        / "images"
        / "train"
        / "vindr_300"
    )

    dicom_files = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in [".dcm", ".dicom"]
    ])

    if not dicom_files:
        raise FileNotFoundError(
            f"No DICOM files found in {IMAGE_DIR}"
        )

    # Initialize model once
    inference = ChestXrayInference()

    # Predict first DICOM
    test_file = dicom_files[0]

    print("\nTesting image:", test_file.name)

    result = inference.predict(test_file)

    print("\nINFERENCE RESULTS")
    print("-----------------")

    for class_name, mask in result["masks"].items():

        print(
            f"{class_name}: "
            f"Shape={mask.shape}, "
            f"Positive pixels={int(mask.sum())}"
        )

    print("\nOriginal image size:", result["original_size"])
    print("Input size:", result["input_size"])
    print("Device:", result["device"])

    print("\nINFERENCE TEST PASSED!")
