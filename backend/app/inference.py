
from pathlib import Path

import torch

from app.model import UNet
from app.preprocessing import preprocess_dicom


# Project paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BACKEND_DIR / "weights" / "best_unet_baseline.pth"

# Model configuration
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


class ChestXrayInference:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        # Initialize the exact trained architecture
        self.model = UNet(
            in_channels=1,
            out_channels=5
        ).to(self.device)

        # Load trained weights
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    @torch.inference_mode()
    def predict(self, dicom_path):

        # Preprocess DICOM
        image_tensor, original_size, ds = preprocess_dicom(
            dicom_path
        )

        image_tensor = image_tensor.to(self.device)

        # Model prediction
        logits = self.model(image_tensor)

        probabilities = torch.sigmoid(logits)

        # Convert probabilities into binary masks
        probability_array = probabilities[0].cpu().numpy()

        masks = {}

        for index, class_name in enumerate(CLASS_NAMES):

            threshold = THRESHOLDS[class_name]

            binary_mask = (
                probability_array[index] >= threshold
            ).astype("uint8")

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


# Standalone test
if __name__ == "__main__":

    PROJECT_ROOT = BACKEND_DIR.parent

    IMAGE_DIR = (
        PROJECT_ROOT
        / "data"
        / "images"
        / "train"
        / "vindr_300"
    )

    dicom_files = [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in [".dcm", ".dicom"]
    ]

    if not dicom_files:
        raise FileNotFoundError("No DICOM files found.")

    inference = ChestXrayInference()

    result = inference.predict(dicom_files[0])

    print("\nINFERENCE RESULTS")
    print("-----------------")

    for class_name, mask in result["masks"].items():

        print(
            f"{class_name}: "
            f"Shape={mask.shape}, "
            f"Positive pixels={int(mask.sum())}"
        )

    print("\nOriginal image size:", result["original_size"])
    print("Device:", result["device"])
    print("\nINFERENCE TEST PASSED!")
