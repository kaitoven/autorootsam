import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from ..utils.timelapse import parse_timestamp_from_image_name, make_sequence_id


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _resolve_path(root_dir: str, filename: str, candidates: List[str]) -> str:
    """Resolve `filename` by trying candidate subdirs under `root_dir`.

    `candidates` are subdirectory strings (e.g. "train/images/Cotton_736x552_DPI150").
    """
    # Absolute or direct path
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    # Try candidates
    for c in candidates:
        p = os.path.join(root_dir, c, filename) if c else os.path.join(root_dir, filename)
        if os.path.exists(p):
            return p

    # Final fallback: direct join (may raise FileNotFoundError later)
    return os.path.join(root_dir, filename)


def flux_from_mask(mask: np.ndarray) -> np.ndarray:
    """Pseudo tangent field from distance-transform gradient (per-frame, not motion)."""
    dist = distance_transform_edt(mask > 0).astype(np.float32)
    if dist.max() < 1e-6:
        return np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
    gy, gx = np.gradient(dist)
    v = np.stack([gx, gy], axis=0)
    n = np.linalg.norm(v, axis=0) + 1e-6
    v = (v / n).astype(np.float32)
    return v


def to_tensor_img(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask).unsqueeze(0).float()


_PRMI_JSON_RE = re.compile(r"^(?P<subset>.+)_(?P<split>train|val|test)\.json$", re.IGNORECASE)


def infer_split_subset_from_json(json_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer (split, subset_dir) from PRMI json filename.

    Expected PRMI naming:
        Cotton_736x552_DPI150_train.json
        Peanut_640x480_DPI120_val.json

    Returns (split, subset) or (None, None) if parsing fails.
    """
    base = os.path.basename(json_path)
    m = _PRMI_JSON_RE.match(base)
    if not m:
        return None, None
    subset = m.group("subset")
    split = m.group("split").lower()
    return split, subset


@dataclass
class PRMIRecord:
    image_name: str
    binary_mask: str
    crop: str
    date: str
    tube_num: str
    depth: str
    dpi: str
    location: str
    has_root: int

    @property
    def seq_id(self) -> str:
        return make_sequence_id(self.crop, self.location, self.tube_num, self.depth, self.dpi)


def load_prmi_json(json_path: str) -> List[PRMIRecord]:
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of PRMI records")

    out: List[PRMIRecord] = []
    for r in data:
        out.append(
            PRMIRecord(
                image_name=str(r["image_name"]),
                binary_mask=str(r["binary_mask"]),
                crop=str(r.get("crop", "")),
                date=str(r.get("date", "")),
                tube_num=str(r.get("tube_num", "")),
                depth=str(r.get("depth", "")),
                dpi=str(r.get("dpi", "")),
                location=str(r.get("location", "")),
                has_root=int(r.get("has_root", 0)),
            )
        )
    return out


def _default_subdirs(root_dir: str, json_path: str, split: Optional[str], subset: Optional[str]) -> Tuple[List[str], List[str]]:
    """Build default search subdirs for images and masks.

    This is aligned with the user's PRMI folder structure:
        PRMI/<split>/images/<subset>/<image_name>
        PRMI/<split>/masks_pixel_gt/<subset>/<binary_mask>
    """
    # If split/subset not given, try inference
    if split is None or subset is None:
        s2, sub2 = infer_split_subset_from_json(json_path)
        split = split or s2
        subset = subset or sub2

    img_candidates: List[str] = []
    msk_candidates: List[str] = []

    if split and subset:
        img_candidates += [
            os.path.join(split, "images", subset),
            os.path.join(split, "images"),
        ]
        msk_candidates += [
            os.path.join(split, "masks_pixel_gt", subset),
            os.path.join(split, "masks_pixel_gt"),
        ]

    # Common fallbacks
    img_candidates += ["images", "JPEGImages", "", "."]
    msk_candidates += ["masks_pixel_gt", "masks", "Annotations", "", "."]

    # Deduplicate while preserving order
    def _uniq(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _uniq(img_candidates), _uniq(msk_candidates)


class PRMIDatasetSingle(Dataset):
    """Single-frame PRMI dataset.

    Notes:
    - `root_dir` should point to the PRMI root directory that contains `train/val/test`.
    - `json_path` is a PRMI split json like `.../Cotton_736x552_DPI150_train.json`.
    """

    def __init__(
        self,
        root_dir: str,
        json_path: Optional[str] = None,
        *,
        json_file: Optional[str] = None,  # alias
        split: Optional[str] = None,
        subset: Optional[str] = None,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        images_subdirs: Optional[List[str]] = None,
        masks_subdirs: Optional[List[str]] = None,
        return_flux: bool = True,
        transform=None,
    ):
        super().__init__()
        json_path = json_path or json_file
        if json_path is None:
            raise ValueError("json_path is required")

        self.root_dir = root_dir
        self.json_path = json_path
        self.split = split
        self.subset = subset
        self.records = load_prmi_json(json_path)

        img_defaults, msk_defaults = _default_subdirs(root_dir, json_path, split, subset)
        self.images_subdirs = []
        self.masks_subdirs = []
        if images_dir:
            self.images_subdirs.append(images_dir)
        if masks_dir:
            self.masks_subdirs.append(masks_dir)
        self.images_subdirs += (images_subdirs or img_defaults)
        self.masks_subdirs += (masks_subdirs or msk_defaults)

        self.return_flux = return_flux
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        img_path = _resolve_path(self.root_dir, rec.image_name, self.images_subdirs)
        mask_path = _resolve_path(self.root_dir, rec.binary_mask, self.masks_subdirs)

        img = _read_image(img_path)
        H, W = img.shape[:2]
        mask = _read_mask(mask_path, (H, W))

        sample = {
            "image": to_tensor_img(img),
            "mask": to_tensor_mask(mask),
            "has_root": torch.tensor([[float(rec.has_root)]], dtype=torch.float32),
            "meta": {
                "seq_id": rec.seq_id,
                "image_name": rec.image_name,
                "binary_mask": rec.binary_mask,
                "date": rec.date,
                "split": self.split,
                "subset": self.subset,
            },
        }
        if self.return_flux:
            sample["flux"] = torch.from_numpy(flux_from_mask(mask))

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class PRMIPairDataset(Dataset):
    """Adjacent time-lapse pairs from the same sequence_id.

    Each item returns (prev, cur) with Δt in days.

    Keys:
        image_prev, mask_prev, has_root_prev
        image, mask, has_root
        delta_t  (B,1)
        (optional) flux_prev, flux
        anchor (if pair_transform adds it)
    """

    def __init__(
        self,
        root_dir: str,
        json_path: Optional[str] = None,
        *,
        json_file: Optional[str] = None,  # alias
        split: Optional[str] = None,
        subset: Optional[str] = None,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        images_subdirs: Optional[List[str]] = None,
        masks_subdirs: Optional[List[str]] = None,
        return_flux: bool = True,
        pair_transform=None,
        # Optional filtering
        min_dt_days: float = 0.0,
        max_dt_days: Optional[float] = None,
    ):
        super().__init__()
        json_path = json_path or json_file
        if json_path is None:
            raise ValueError("json_path is required")

        self.root_dir = root_dir
        self.json_path = json_path
        self.split = split
        self.subset = subset
        self.records = load_prmi_json(json_path)

        img_defaults, msk_defaults = _default_subdirs(root_dir, json_path, split, subset)
        self.images_subdirs = []
        self.masks_subdirs = []
        if images_dir:
            self.images_subdirs.append(images_dir)
        if masks_dir:
            self.masks_subdirs.append(masks_dir)
        self.images_subdirs += (images_subdirs or img_defaults)
        self.masks_subdirs += (masks_subdirs or msk_defaults)

        self.return_flux = return_flux
        self.pair_transform = pair_transform

        # Build adjacent pairs per sequence
        by_seq: Dict[str, List[int]] = {}
        timestamps: List[Optional[object]] = []
        for i, r in enumerate(self.records):
            ts = parse_timestamp_from_image_name(r.image_name, r.date)
            timestamps.append(ts)
            by_seq.setdefault(r.seq_id, []).append(i)

        pairs: List[Tuple[int, int, float]] = []
        for _seq, idxs in by_seq.items():
            idxs = sorted(idxs, key=lambda j: (timestamps[j] is None, timestamps[j]))
            for a, b in zip(idxs[:-1], idxs[1:]):
                ta, tb = timestamps[a], timestamps[b]
                if ta is None or tb is None:
                    continue
                dt_days = (tb - ta).total_seconds() / 86400.0
                if dt_days <= min_dt_days:
                    continue
                if max_dt_days is not None and dt_days > max_dt_days:
                    continue
                pairs.append((a, b, float(dt_days)))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        ia, ib, dt_days = self.pairs[idx]
        ra, rb = self.records[ia], self.records[ib]

        img_path_a = _resolve_path(self.root_dir, ra.image_name, self.images_subdirs)
        mask_path_a = _resolve_path(self.root_dir, ra.binary_mask, self.masks_subdirs)
        img_path_b = _resolve_path(self.root_dir, rb.image_name, self.images_subdirs)
        mask_path_b = _resolve_path(self.root_dir, rb.binary_mask, self.masks_subdirs)

        img_a = _read_image(img_path_a)
        img_b = _read_image(img_path_b)

        H, W = img_a.shape[:2]
        if img_b.shape[:2] != (H, W):
            img_b = cv2.resize(img_b, (W, H), interpolation=cv2.INTER_LINEAR)

        mask_a = _read_mask(mask_path_a, (H, W))
        mask_b = _read_mask(mask_path_b, (H, W))

        sample = {
            "image_prev": to_tensor_img(img_a),
            "mask_prev": to_tensor_mask(mask_a),
            "has_root_prev": torch.tensor([[float(ra.has_root)]], dtype=torch.float32),
            "image": to_tensor_img(img_b),
            "mask": to_tensor_mask(mask_b),
            "has_root": torch.tensor([[float(rb.has_root)]], dtype=torch.float32),
            "delta_t": torch.tensor([[dt_days]], dtype=torch.float32),
            "meta": {
                "seq_id": rb.seq_id,
                "image_prev": ra.image_name,
                "image": rb.image_name,
                "dt_days": dt_days,
                "split": self.split,
                "subset": self.subset,
            },
        }
        if self.return_flux:
            sample["flux_prev"] = torch.from_numpy(flux_from_mask(mask_a))
            sample["flux"] = torch.from_numpy(flux_from_mask(mask_b))

        if self.pair_transform is not None:
            sample = self.pair_transform(sample)
        return sample
