#!/usr/bin/env python3
"""AutoDL sanity check for PRMI layout + JSON ↔ file paths.

This script is intentionally dependency-light (stdlib only).

It validates that:
  1) The split inferred from JSON path exists under root_dir.
  2) The subset folder exists under images/ and masks_pixel_gt/.
  3) A sample of entries in the JSON can be resolved to existing image/mask files.

Usage:
  python tools/autodl_sanity_check.py --root_dir /root/autodl-tmp/PRMI \
    --json /root/autodl-tmp/PRMI/train/labels_image_gt/Cotton_736x552_DPI150_train.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


SPLITS = ("train", "val", "test")


def _infer_split_from_json_path(json_path: str) -> Optional[str]:
    parts = os.path.normpath(json_path).split(os.sep)
    for s in SPLITS:
        if s in parts:
            return s
    return None


def _infer_subset_from_json_name(json_path: str) -> str:
    base = os.path.basename(json_path)
    for suf in ("_train.json", "_val.json", "_test.json"):
        if base.endswith(suf):
            return base[: -len(suf)]
    if base.endswith(".json"):
        return base[:-5]
    return base


def _mask_name_from_image_name(image_name: str) -> str:
    # PRMI convention: GT_<image_basename>.png
    # Example: Cotton_T004_L016_2012.06.22_091757_AMC_DPI150.jpg
    #       -> GT_Cotton_T004_L016_2012.06.22_091757_AMC_DPI150.png
    stem = os.path.splitext(os.path.basename(image_name))[0]
    return f"GT_{stem}.png"


@dataclass
class CheckResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


def _load_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some exporters wrap in {"images": [...]} etc.
        for k in ("images", "data", "items", "annotations"):
            if k in data and isinstance(data[k], list):
                return data[k]
        # If dict of id->item
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        raise ValueError("Unsupported JSON dict format")
    if not isinstance(data, list):
        raise ValueError("JSON must be a list (or a supported dict wrapper)")
    return data


def sanity_check(root_dir: str, json_path: str, sample_n: int, seed: int) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    if not os.path.isdir(root_dir):
        errors.append(f"root_dir not found: {root_dir}")
        return CheckResult(False, errors, warnings, details)
    if not os.path.isfile(json_path):
        errors.append(f"json not found: {json_path}")
        return CheckResult(False, errors, warnings, details)

    split = _infer_split_from_json_path(json_path)
    if split is None:
        errors.append(
            f"Could not infer split from json path. Expected one of {SPLITS} in path: {json_path}"
        )
        return CheckResult(False, errors, warnings, details)

    subset = _infer_subset_from_json_name(json_path)
    details["split"] = split
    details["subset"] = subset

    img_dir = os.path.join(root_dir, split, "images", subset)
    msk_dir = os.path.join(root_dir, split, "masks_pixel_gt", subset)
    lbl_dir = os.path.join(root_dir, split, "labels_image_gt")

    for p in (os.path.join(root_dir, split), lbl_dir):
        if not os.path.isdir(p):
            errors.append(f"Missing directory: {p}")

    if not os.path.isdir(img_dir):
        errors.append(f"Missing images subset dir: {img_dir}")
    if not os.path.isdir(msk_dir):
        warnings.append(
            f"Missing masks subset dir (training/eval with GT will fail): {msk_dir}"
        )

    if errors:
        return CheckResult(False, errors, warnings, details)

    items = _load_json(json_path)
    details["num_items"] = len(items)
    if len(items) == 0:
        errors.append("JSON has 0 items")
        return CheckResult(False, errors, warnings, details)

    # Determine which key stores image filename.
    key_candidates = ("image_name", "file_name", "image", "filename")
    img_key = None
    for k in key_candidates:
        if k in items[0]:
            img_key = k
            break
    if img_key is None:
        errors.append(f"Could not find image filename key in JSON (tried {key_candidates}).")
        return CheckResult(False, errors, warnings, details)
    details["image_key"] = img_key

    # Sample a set of entries (deterministic).
    rnd = random.Random(seed)
    idxs = list(range(len(items)))
    rnd.shuffle(idxs)
    idxs = idxs[: min(sample_n, len(idxs))]

    missing_images: List[str] = []
    missing_masks: List[str] = []
    bad_masks_key: int = 0

    for i in idxs:
        it = items[i]
        image_name = it.get(img_key)
        if not image_name:
            errors.append(f"Item[{i}] missing '{img_key}'")
            continue
        image_path = os.path.join(img_dir, os.path.basename(image_name))
        if not os.path.isfile(image_path):
            missing_images.append(image_path)

        # mask can be explicit in JSON, but we also validate PRMI naming convention.
        mask_name = _mask_name_from_image_name(image_name)
        mask_path = os.path.join(msk_dir, mask_name)
        if os.path.isdir(msk_dir) and not os.path.isfile(mask_path):
            missing_masks.append(mask_path)

        # if JSON has a mask field, check it too
        for mk in ("mask_name", "mask", "binary_mask", "gt_mask"):
            if mk in it and it[mk]:
                bad = False
                v = it[mk]
                # many JSONs store just a filename
                if isinstance(v, str):
                    p2 = os.path.join(msk_dir, os.path.basename(v))
                    if os.path.isdir(msk_dir) and not os.path.isfile(p2):
                        bad = True
                # some store dicts
                elif isinstance(v, dict) and "file_name" in v:
                    p2 = os.path.join(msk_dir, os.path.basename(v["file_name"]))
                    if os.path.isdir(msk_dir) and not os.path.isfile(p2):
                        bad = True
                if bad:
                    bad_masks_key += 1
                break

    details["checked_samples"] = len(idxs)
    details["missing_images"] = len(missing_images)
    details["missing_masks"] = len(missing_masks)
    details["bad_masks_key_samples"] = bad_masks_key

    if missing_images:
        errors.append(
            "Missing image files (showing up to 5):\n" + "\n".join(missing_images[:5])
        )
    if missing_masks:
        warnings.append(
            "Missing GT mask files (showing up to 5):\n" + "\n".join(missing_masks[:5])
        )
    if bad_masks_key > 0:
        warnings.append(
            f"Some samples reference mask fields that do not exist on disk (count in sample={bad_masks_key})."
        )

    ok = len(errors) == 0
    return CheckResult(ok, errors, warnings, details)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--sample", type=int, default=64, help="how many JSON items to sample")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    res = sanity_check(args.root_dir, args.json, args.sample, args.seed)
    print(json.dumps(res.details, indent=2, ensure_ascii=False))
    if res.warnings:
        print("\n[WARN]")
        for w in res.warnings:
            print("-", w)
    if res.errors:
        print("\n[ERROR]", file=sys.stderr)
        for e in res.errors:
            print("-", e, file=sys.stderr)
    sys.exit(0 if res.ok else 2)


if __name__ == "__main__":
    main()
