"""Project-wide constants.

Centralizing these guarantees that the dataset parser, the split generator, the
dataloaders, and the model all agree on the same label / magnification encodings.
Nothing here should be duplicated elsewhere in the codebase.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Class encoding (binary task: benign vs malignant)
# --------------------------------------------------------------------------- #
# The BreaKHis directory splits samples into `benign` and `malignant`; the
# filename token `B` / `M` encodes the same thing. We treat malignant as the
# positive class (1), which is the convention for medical screening tasks.
CLASS_NAMES = ("benign", "malignant")
CLASS_TO_IDX = {"benign": 0, "malignant": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Filename tumor-class token -> canonical class name.
FILENAME_CLASS_TOKEN = {"B": "benign", "M": "malignant"}

# --------------------------------------------------------------------------- #
# Magnification encoding
# --------------------------------------------------------------------------- #
# One unified model is trained across every magnification. The magnification is
# exposed as metadata on every sample and encoded as a contiguous index so that
# it can drive an optional magnification-embedding module.
MAGNIFICATIONS = (40, 100, 200, 400)
MAG_TO_IDX = {mag: i for i, mag in enumerate(MAGNIFICATIONS)}
IDX_TO_MAG = {i: mag for mag, i in MAG_TO_IDX.items()}
NUM_MAGNIFICATIONS = len(MAGNIFICATIONS)

# --------------------------------------------------------------------------- #
# Dataset protocol
# --------------------------------------------------------------------------- #
# BreaKHis contains 82 unique patients (24 benign, 58 malignant). The protocol
# permanently holds out 16 patients for the final test set and runs 5-fold
# patient-level cross validation on the remaining 66.
TOTAL_PATIENTS = 82
NUM_TEST_PATIENTS = 16
NUM_CV_PATIENTS = TOTAL_PATIENTS - NUM_TEST_PATIENTS  # 66
DEFAULT_NUM_FOLDS = 5

IMAGE_EXTENSIONS = (".png",)
