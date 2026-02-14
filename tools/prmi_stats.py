import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np

from mdrs.data.prmi_dataset import load_prmi_json
from mdrs.utils.timelapse import parse_timestamp_from_image_name, make_sequence_id, delta_days


def _percentiles(x: np.ndarray, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": float('nan') for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}


def _median_iqr(x: np.ndarray) -> Tuple[float, float, float]:
    if x.size == 0:
        return float('nan'), float('nan'), float('nan')
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(q1), float(q3)


def analyze_jsons(json_paths: List[str]) -> Dict:
    # Load all records
    records = []
    for jp in json_paths:
        rs = load_prmi_json(jp)
        for r in rs:
            r._source_json = os.path.basename(jp)  # type: ignore
        records.extend(rs)

    out: Dict = {}
    out["num_images"] = len(records)
    out["num_root"] = int(sum(int(r.has_root) for r in records))
    out["num_non_root"] = out["num_images"] - out["num_root"]
    out["root_ratio"] = float(out["num_root"] / max(1, out["num_images"]))

    # Unique meta
    out["num_locations"] = int(len({r.location for r in records}))
    out["num_tubes"] = int(len({r.tube_num for r in records}))
    out["num_depths"] = int(len({r.depth for r in records}))
    out["num_subsets"] = int(len({r.subset for r in records}))

    # Build sequences
    by_seq = defaultdict(list)
    for r in records:
        ts = parse_timestamp_from_image_name(r.image_name, r.date)
        if ts is None:
            continue
        sid = make_sequence_id(r.crop, r.location, r.tube_num, r.depth, r.dpi)
        by_seq[sid].append((ts, r))

    seq_lens = []
    deltas = []

    for sid, items in by_seq.items():
        items.sort(key=lambda x: x[0])
        seq_lens.append(len(items))
        for (t0, _), (t1, _) in zip(items[:-1], items[1:]):
            deltas.append(delta_days(t0, t1))

    seq_lens_np = np.asarray(seq_lens, dtype=np.float32)
    deltas_np = np.asarray(deltas, dtype=np.float32)

    out["num_sequences"] = int(len(by_seq))
    out["single_frame_seq_ratio"] = float(np.mean(seq_lens_np == 1)) if seq_lens_np.size else float('nan')
    out["seq_len_percentiles"] = _percentiles(seq_lens_np)
    out["dt_days_percentiles"] = _percentiles(deltas_np)

    # Per-subset breakdown
    per_subset = {}
    by_subset = defaultdict(list)
    for r in records:
        by_subset[r.subset].append(r)

    for subset, rs in sorted(by_subset.items()):
        # sequences within subset
        by_seq_s = defaultdict(list)
        deltas_s = []
        seq_lens_s = []
        root_s = int(sum(int(r.has_root) for r in rs))
        for r in rs:
            ts = parse_timestamp_from_image_name(r.image_name, r.date)
            if ts is None:
                continue
            sid = make_sequence_id(r.crop, r.location, r.tube_num, r.depth, r.dpi)
            by_seq_s[sid].append((ts, r))
        for sid, items in by_seq_s.items():
            items.sort(key=lambda x: x[0])
            seq_lens_s.append(len(items))
            for (t0, _), (t1, _) in zip(items[:-1], items[1:]):
                deltas_s.append(delta_days(t0, t1))

        seq_lens_s = np.asarray(seq_lens_s, dtype=np.float32)
        deltas_s = np.asarray(deltas_s, dtype=np.float32)

        per_subset[subset] = {
            "num_images": int(len(rs)),
            "num_root": int(root_s),
            "num_non_root": int(len(rs) - root_s),
            "root_ratio": float(root_s / max(1, len(rs))),
            "num_sequences": int(len(by_seq_s)),
            "single_frame_seq_ratio": float(np.mean(seq_lens_s == 1)) if seq_lens_s.size else float('nan'),
            "seq_len_percentiles": _percentiles(seq_lens_s),
            "dt_days_percentiles": _percentiles(deltas_s),
            "num_locations": int(len({r.location for r in rs})),
            "num_tubes": int(len({r.tube_num for r in rs})),
            "num_depths": int(len({r.depth for r in rs})),
            "dpi_values": sorted({r.dpi for r in rs}),
            "shape_hw_values": sorted({tuple(r.shape_hw) for r in rs}),
        }

    out["per_subset"] = per_subset

    # Per-split based on json name suffix (best-effort)
    per_split = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    for r in records:
        src = getattr(r, "_source_json", "")
        if src.endswith("_train.json"):
            per_split["train"] += 1
        elif src.endswith("_val.json"):
            per_split["val"] += 1
        elif src.endswith("_test.json"):
            per_split["test"] += 1
        else:
            per_split["unknown"] += 1
    out["per_split"] = per_split

    return out


def main():
    ap = argparse.ArgumentParser(description="Compute PRMI (no Switchgrass) time-lapse statistics from split json files")
    ap.add_argument("--json", action="append", default=[], help="Path to a PRMI split json. Can be passed multiple times.")
    ap.add_argument("--json_dir", default="", help="If set, auto-add all *_train.json, *_val.json, *_test.json under this dir")
    ap.add_argument("--out", default="", help="Optional output path (.json)")
    args = ap.parse_args()

    jsons = list(args.json)
    if args.json_dir:
        for fn in sorted(os.listdir(args.json_dir)):
            if fn.endswith(".json") and (fn.endswith("_train.json") or fn.endswith("_val.json") or fn.endswith("_test.json")):
                jsons.append(os.path.join(args.json_dir, fn))

    if not jsons:
        raise SystemExit("No json provided. Use --json ... or --json_dir ...")

    stats = analyze_jsons(jsons)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    # Pretty print a concise summary
    print("=== PRMI stats (from provided jsons) ===")
    print(f"images: {stats['num_images']} | root: {stats['num_root']} | non-root: {stats['num_non_root']} | root_ratio: {stats['root_ratio']:.4f}")
    print(f"sequences: {stats['num_sequences']} | single-frame seq ratio: {stats['single_frame_seq_ratio']:.4f}")
    print("dt_days percentiles:", stats["dt_days_percentiles"])
    print("seq_len percentiles:", stats["seq_len_percentiles"])
    print("per_split:", stats["per_split"])
    print("subsets:")
    for k, v in stats["per_subset"].items():
        print(f" - {k}: images={v['num_images']} root_ratio={v['root_ratio']:.3f} seqs={v['num_sequences']} single={v['single_frame_seq_ratio']:.3f} dt_p50={v['dt_days_percentiles']['p50']:.2f}")


if __name__ == "__main__":
    main()
