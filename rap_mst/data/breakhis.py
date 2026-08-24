"""BreaKHis filesystem parsing and metadata extraction.

This module is the single source of truth for turning the raw BreaKHis directory
into structured, typed records. Both the split generator (`scripts/prepare_splits.py`)
and the runtime `Dataset` rely on it, which guarantees that patient-id extraction
is *identical* everywhere -- the property the whole no-leakage protocol depends on.

BreaKHis filename grammar
-------------------------
Example: ``SOB_B_A-14-22549AB-100-001.png``

    SOB          biopsy procedure (Surgical Open Biopsy)
    B            tumor class token      (B = benign, M = malignant)
    A            tumor type token       (A = adenosis, DC = ductal carcinoma, ...)
    14-22549AB   patient / slide id     (year-slide)
    100          magnification          (40 | 100 | 200 | 400)
    001          image sequence number

The canonical, globally-unique ``patient_id`` is the biopsy+class+type+slide
prefix, e.g. ``SOB_B_A-14-22549AB``. There are exactly 82 such ids, matching the
82 patient folders on disk, so the folder and the parsed id map 1:1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from rap_mst.constants import (
    CLASS_TO_IDX,
    FILENAME_CLASS_TOKEN,
    IMAGE_EXTENSIONS,
    MAG_TO_IDX,
)

# SOB_<class>_<type>-<year>-<slide>-<mag>-<seq>.png
_FILENAME_RE = re.compile(
    r"^(?P<procedure>[A-Z]+)_(?P<cls>[A-Z]+)_(?P<ttype>[A-Z]+)-"
    r"(?P<year>\d+)-(?P<slide>[0-9A-Za-z]+)-"
    r"(?P<mag>\d+)-(?P<seq>\d+)$"
)


@dataclass(frozen=True)
class Sample:
    """One image and all metadata future modules might need.

    Every field that a downstream retrieval / prototype module could plausibly
    key on is captured here so those modules never have to re-parse the dataset.
    """

    image_path: str
    patient_id: str
    label: int          # 0 = benign, 1 = malignant
    class_name: str
    tumor_type: str     # filename token, e.g. "A", "DC"
    magnification: int  # raw value: 40 / 100 / 200 / 400
    mag_index: int      # contiguous index used by the magnification embedding
    seq: int            # image sequence number within the slide

    def as_dict(self) -> Dict:
        return asdict(self)


def parse_filename(path: Path) -> Sample:
    """Parse a single BreaKHis image path into a :class:`Sample`.

    Raises
    ------
    ValueError
        If the filename does not match the BreaKHis grammar or carries an
        unexpected class / magnification token. Failing loudly here is
        intentional: silent mis-parsing would corrupt the patient splits.
    """
    stem = path.stem
    m = _FILENAME_RE.match(stem)
    if m is None:
        raise ValueError(f"Filename does not match BreaKHis grammar: {path}")

    cls_token = m.group("cls")
    if cls_token not in FILENAME_CLASS_TOKEN:
        raise ValueError(f"Unknown class token {cls_token!r} in {path}")
    class_name = FILENAME_CLASS_TOKEN[cls_token]

    magnification = int(m.group("mag"))
    if magnification not in MAG_TO_IDX:
        raise ValueError(f"Unexpected magnification {magnification} in {path}")

    patient_id = (
        f"{m.group('procedure')}_{cls_token}_{m.group('ttype')}"
        f"-{m.group('year')}-{m.group('slide')}"
    )

    return Sample(
        image_path=str(path),
        patient_id=patient_id,
        label=CLASS_TO_IDX[class_name],
        class_name=class_name,
        tumor_type=m.group("ttype"),
        magnification=magnification,
        mag_index=MAG_TO_IDX[magnification],
        seq=int(m.group("seq")),
    )


def subtype_from_patient_id(patient_id: str) -> str:
    """``'SOB_M_DC-14-12312'`` -> ``'DC'``.

    Patient ids are produced by :func:`parse_filename`, so the tumor-type token is
    always the third underscore-separated field. Kept here (never re-implemented
    downstream) so the retrieval bank's ``subtype`` column and the split file's
    ``tumor_type`` can never disagree.
    """
    try:
        return patient_id.split("_", 2)[2].split("-", 1)[0]
    except IndexError:  # pragma: no cover - defensive; ids come from parse_filename
        raise ValueError(f"Malformed patient id (expected SOB_<cls>_<type>-...): {patient_id!r}")


def scan_dataset(root: str | Path) -> List[Sample]:
    """Recursively scan the BreaKHis root and return every parsed sample.

    ``root`` may point at the dataset top level (``BreaKHis_v1``) or any parent
    directory containing the images; every ``*.png`` under it is parsed.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"BreaKHis root does not exist: {root}")

    samples: List[Sample] = []
    for ext in IMAGE_EXTENSIONS:
        for img_path in sorted(root.rglob(f"*{ext}")):
            samples.append(parse_filename(img_path))

    if not samples:
        raise RuntimeError(f"No images found under {root}. Check the dataset path.")
    return samples


def patient_index(samples: List[Sample]) -> Dict[str, Dict]:
    """Aggregate samples per patient.

    Returns a mapping ``patient_id -> {label, class_name, tumor_type, num_images}``.
    Because a patient is entirely benign or entirely malignant in BreaKHis (and a
    single tumor type), both label and tumor type are well-defined at the patient
    level -- exactly what the stratified split needs. ``tumor_type`` is carried so
    splitting can stratify by the eight fine-grained subtypes, not just the binary
    class.
    """
    index: Dict[str, Dict] = {}
    for s in samples:
        entry = index.setdefault(
            s.patient_id,
            {
                "label": s.label,
                "class_name": s.class_name,
                "tumor_type": s.tumor_type,
                "num_images": 0,
            },
        )
        if entry["label"] != s.label:
            raise ValueError(
                f"Patient {s.patient_id} has mixed labels -- dataset assumption violated."
            )
        if entry["tumor_type"] != s.tumor_type:
            raise ValueError(
                f"Patient {s.patient_id} has mixed tumor types -- dataset assumption violated."
            )
        entry["num_images"] += 1
    return index
