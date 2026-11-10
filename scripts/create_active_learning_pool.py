
from pathlib import Path

import pandas as pd
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS_CSV = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "image_labels_train.csv"
)

EXISTING_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vindr_300_image_manifest.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_MANIFEST = (
    OUTPUT_DIR / "active_learning_pool_manifest.csv"
)

POOL_SIZE = 50
RANDOM_SEED = 2026

BASE_URL = (
    "https://physionet.org/files/"
    "vindr-cxr/1.0.0/train/"
)


# ==========================================
# 1. LOAD IMAGE IDS
# ==========================================

print("Loading VinDr-CXR image IDs...")

labels_df = pd.read_csv(LABELS_CSV)

if "image_id" not in labels_df.columns:
    raise ValueError("image_id column not found in labels CSV.")

all_image_ids = (
    labels_df["image_id"]
    .dropna()
    .astype(str)
    .unique()
)

print(f"Available unique images: {len(all_image_ids)}")


# ==========================================
# 2. LOAD EXISTING 300 IMAGE IDS
# ==========================================

print("\nLoading existing dataset manifest...")

existing_df = pd.read_csv(EXISTING_MANIFEST)

if "image_id" not in existing_df.columns:
    raise ValueError(
        "image_id column not found in existing manifest."
    )

existing_ids = set(
    existing_df["image_id"]
    .dropna()
    .astype(str)
)

print(f"Existing image IDs: {len(existing_ids)}")


# ==========================================
# 3. EXCLUDE EXISTING IMAGES
# ==========================================

remaining_ids = sorted(
    set(all_image_ids) - existing_ids
)

print(f"Remaining eligible images: {len(remaining_ids)}")

if len(remaining_ids) < POOL_SIZE:
    raise ValueError(
        "Not enough unique images for acquisition pool."
    )


# ==========================================
# 4. REPRODUCIBLE RANDOM SAMPLING
# ==========================================

rng = np.random.default_rng(RANDOM_SEED)

selected_ids = rng.choice(
    remaining_ids,
    size=POOL_SIZE,
    replace=False
)


# ==========================================
# 5. CREATE ACQUISITION MANIFEST
# ==========================================

pool_df = pd.DataFrame({
    "image_id": selected_ids,
})

pool_df["filename"] = (
    pool_df["image_id"] + ".dicom"
)

pool_df["source_url"] = (
    BASE_URL + pool_df["filename"]
)

pool_df["status"] = "pending_download"

pool_df["annotation_status"] = "unreviewed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

pool_df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


# ==========================================
# 6. VALIDATE NO OVERLAP
# ==========================================

assert len(pool_df) == POOL_SIZE

assert pool_df["image_id"].nunique() == POOL_SIZE

assert not (
    set(pool_df["image_id"]) & existing_ids
)

print("\n===================================")
print("ACTIVE LEARNING POOL CREATED")
print("===================================")

print(f"Pool size: {len(pool_df)}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Overlap with existing 300: 0")

print(f"\nSaved manifest:\n{OUTPUT_MANIFEST}")

print("\nFirst five selected images:")
print(pool_df[["image_id", "filename"]].head())

print("\nPOOL GENERATION SUCCESSFUL!")
