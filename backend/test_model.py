
import torch
from app.model import UNet

CHECKPOINT_PATH = "weights/best_unet_baseline.pth"

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize exact trained architecture
model = UNet(
    in_channels=1,
    num_classes=5
).to(device)

# Load trained checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Test with a dummy X-ray tensor
dummy_input = torch.randn(
    1, 1, 256, 256
).to(device)

with torch.no_grad():
    logits = model(dummy_input)
    probabilities = torch.sigmoid(logits)

print("\nMODEL LOADED SUCCESSFULLY!")
print("Device:", device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Output shape:", logits.shape)

print(
    "Probability range:",
    probabilities.min().item(),
    probabilities.max().item()
)

assert logits.shape == (1, 5, 256, 256)

print("\nCHECKPOINT VALIDATION PASSED!")
