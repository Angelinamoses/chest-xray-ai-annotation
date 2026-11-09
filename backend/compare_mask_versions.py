
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

OLD_MASK_DIR = PROCESSED / "masks_256"
NEW_MASK_DIR = PROCESSED / "masks_256_audited"

OUTPUT_DIR = (
    ROOT / "backend" / "outputs" / "mask_version_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Aortic enlargement",
    "Cardiomegaly",
    "Pleural thickening",
    "Pulmonary fibrosis",
    "Lung Opacity",
]

rows = []

old_files = {p.stem for p in OLD_MASK_DIR.glob("*.npy")}
new_files = {p.stem for p in NEW_MASK_DIR.glob("*.npy")}

all_ids = sorted(old_files | new_files)

print("Original masks:", len(old_files))
print("Audited masks:", len(new_files))
print("Common images:", len(old_files & new_files))

if old_files != new_files:
    print("WARNING: Image IDs differ between versions.")
    print("Missing from audited:", len(old_files - new_files))
    print("Missing from original:", len(new_files - old_files))

for image_id in all_ids:

    old_path = OLD_MASK_DIR / f"{image_id}.npy"
    new_path = NEW_MASK_DIR / f"{image_id}.npy"

    if not old_path.exists() or not new_path.exists():
        continue

    old = np.load(old_path)
    new = np.load(new_path)

    if old.shape != new.shape:
        print("SHAPE MISMATCH:", image_id, old.shape, new.shape)
        continue

    for channel, class_name in enumerate(CLASSES):

        a = old[:, :, channel] > 0
        b = new[:, :, channel] > 0

        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        changed = np.logical_xor(a, b).sum()

        dice = (
            2 * intersection / (a.sum() + b.sum())
            if (a.sum() + b.sum()) > 0
            else 1.0
        )

        rows.append({
            "image_id": image_id,
            "class_name": class_name,
            "old_pixels": int(a.sum()),
            "new_pixels": int(b.sum()),
            "changed_pixels": int(changed),
            "dice_agreement": float(dice),
            "identical": bool(np.array_equal(a, b)),
        })

df = pd.DataFrame(rows)

report_path = OUTPUT_DIR / "mask_comparison.csv"
df.to_csv(report_path, index=False)

print("\nCLASS-WISE COMPARISON")

summary = (
    df.groupby("class_name")
    .agg(
        images=("image_id", "nunique"),
        identical_images=("identical", "sum"),
        total_changed_pixels=("changed_pixels", "sum"),
        mean_dice_agreement=("dice_agreement", "mean"),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print("\nOverall identical image-class pairs:",
      int(df["identical"].sum()), "/", len(df))

print("\nReport saved:", report_path)
