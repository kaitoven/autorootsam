#!/usr/bin/env bash
set -euo pipefail

# Root-STC-SAM2.1++-TL — AutoDL one-click runner
#
# Goals:
#   1) Zero-friction path alignment with AutoDL:
#        /root/autodl-tmp/autorootsam  (project)
#        /root/autodl-tmp/PRMI                (dataset)
#   2) Uses SAM2's built-in config path (no local yaml needed):
#        configs/sam2.1/sam2.1_hiera_l.yaml
#   3) Provides modes: check / stats / warmup / tl / infer / infer_seq / all
#
# Quick examples:
#   bash run_autodl_paths.sh --mode check --subset Cotton_736x552_DPI150
#   bash run_autodl_paths.sh --mode warmup --subset Cotton_736x552_DPI150 --epochs 10
#   bash run_autodl_paths.sh --mode tl --subset Cotton_736x552_DPI150 --epochs 30
#   bash run_autodl_paths.sh --mode infer --subset Cotton_736x552_DPI150 --image <path.jpg>
#   bash run_autodl_paths.sh --mode infer_seq --subset Cotton_736x552_DPI150

MODE="check"
SUBSET="Cotton_736x552_DPI150"
SPLIT="train"
GPU="0"
TILE="768"
OVERLAP="128"
BATCH_WARMUP="8"
BATCH_TL="4"
EPOCHS_WARMUP="10"
EPOCHS_TL="30"
NUM_WORKERS="8"
AMP="1"
TAU_DAYS="14.0"
SAM2_IMAGE_SIZE="1024"

# Optional overrides
PROJ_DEFAULT="$(cd "$(dirname "$0")" && pwd)"
DATA_DEFAULT="/root/autodl-tmp/PRMI"
PROJ="$PROJ_DEFAULT"
DATA="$DATA_DEFAULT"
CKPT=""
CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
IMAGE=""

usage() {
  cat <<EOF
Usage: bash run_autodl_paths.sh [args]

Args:
  --mode        check|stats|warmup|tl|infer|infer_seq|all   (default: $MODE)
  --subset      PRMI subset folder name                     (default: $SUBSET)
  --split       train|val|test (for --mode infer default)    (default: $SPLIT)
  --gpu         CUDA device id                              (default: $GPU)
  --proj        project root                                (default: $PROJ)
  --data        PRMI root_dir                               (default: $DATA)
  --ckpt        sam2.1 checkpoint path                       (default: <proj>/checkpoints/sam2.1_hiera_large.pt)
  --cfg         sam2 cfg path (built-in)                     (default: $CFG)

Training:
  --tile        tile size                                   (default: $TILE)
  --batch_warm  batch size for warmup                        (default: $BATCH_WARMUP)
  --batch_tl    batch size for timelapse                      (default: $BATCH_TL)
  --epochs_warm warmup epochs                                 (default: $EPOCHS_WARMUP)
  --epochs_tl   timelapse epochs                               (default: $EPOCHS_TL)
  --num_workers dataloader workers                             (default: $NUM_WORKERS)
  --amp         1 to enable AMP, 0 to disable                 (default: $AMP)
  --tau_days    TL decay constant (days)                      (default: $TAU_DAYS)
  --sam2_image_size sam2 internal image size                  (default: $SAM2_IMAGE_SIZE)

Inference:
  --overlap     overlap for tiled inference                   (default: $OVERLAP)
  --image       image path for --mode infer                   (default: auto-pick first from JSON)

Examples:
  bash run_autodl_paths.sh --mode check --subset Cotton_736x552_DPI150
  bash run_autodl_paths.sh --mode stats --subset Cotton_736x552_DPI150
  bash run_autodl_paths.sh --mode warmup --subset Cotton_736x552_DPI150 --epochs_warm 10
  bash run_autodl_paths.sh --mode tl --subset Cotton_736x552_DPI150 --epochs_tl 30
  bash run_autodl_paths.sh --mode infer --subset Cotton_736x552_DPI150 --split test
  bash run_autodl_paths.sh --mode infer_seq --subset Cotton_736x552_DPI150
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --subset) SUBSET="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --proj) PROJ="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --cfg) CFG="$2"; shift 2;;
    --tile) TILE="$2"; shift 2;;
    --overlap) OVERLAP="$2"; shift 2;;
    --batch_warm) BATCH_WARMUP="$2"; shift 2;;
    --batch_tl) BATCH_TL="$2"; shift 2;;
    --epochs_warm) EPOCHS_WARMUP="$2"; shift 2;;
    --epochs_tl) EPOCHS_TL="$2"; shift 2;;
    --num_workers) NUM_WORKERS="$2"; shift 2;;
    --amp) AMP="$2"; shift 2;;
    --tau_days) TAU_DAYS="$2"; shift 2;;
    --sam2_image_size) SAM2_IMAGE_SIZE="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$CKPT" ]]; then
  CKPT="$PROJ/checkpoints/sam2.1_hiera_large.pt"
fi

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

_log() { echo -e "\n[run_autodl_paths] $*"; }

_need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 2; }
}

_python_ok() {
  python - <<'PY'
import sys
print(sys.version)
PY
}

_check_sam2() {
  python - <<'PY'
try:
  import sam2  # noqa
  import importlib
  importlib.import_module('sam2')
  print('[OK] sam2 is installed')
except Exception as e:
  print('[FAIL] sam2 import failed:', e)
  raise SystemExit(2)
PY
}

_locate_cfg() {
  # If CFG exists as a file relative to PROJ, accept.
  if [[ -f "$PROJ/$CFG" ]]; then
    echo "$PROJ/$CFG"
    return 0
  fi
  # Otherwise, attempt to locate inside installed sam2 package.
  python - <<PY
import os, importlib.util
cfg_rel = "${CFG}"
spec = importlib.util.find_spec('sam2')
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(2)
root = list(spec.submodule_search_locations)[0]
candidate = os.path.join(os.path.dirname(root), cfg_rel)  # sam2 is .../site-packages/sam2
if os.path.isfile(candidate):
    print(candidate)
else:
    # Some installs place configs under the sam2 package root
    candidate2 = os.path.join(root, cfg_rel)
    if os.path.isfile(candidate2):
        print(candidate2)
    else:
        print('')
PY
}

_infer_json_path() {
  local split="$1"
  echo "$DATA/$split/labels_image_gt/${SUBSET}_${split}.json" | sed 's/_val\.json/_val.json/'
}

_infer_json_path_exact() {
  local split="$1"
  case "$split" in
    train) echo "$DATA/train/labels_image_gt/${SUBSET}_train.json";;
    val)   echo "$DATA/val/labels_image_gt/${SUBSET}_val.json";;
    test)  echo "$DATA/test/labels_image_gt/${SUBSET}_test.json";;
    *) echo "";;
  esac
}

_pick_first_image_from_json() {
  local json="$1"
  python - <<PY
import json
import os
path = "${json}"
data = json.load(open(path,'r',encoding='utf-8'))
if isinstance(data, dict):
    for k in ('images','data','items','annotations'):
        if k in data and isinstance(data[k], list):
            data = data[k]
            break
    else:
        data = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
if not isinstance(data, list) or not data:
    raise SystemExit(2)
it = data[0]
for k in ('image_name','file_name','image','filename'):
    if k in it:
        print(os.path.basename(it[k]))
        raise SystemExit(0)
raise SystemExit(2)
PY
}

_check_paths() {
  _log "Checking environment + paths"
  _need python
  _python_ok
  _check_sam2

  if [[ ! -d "$PROJ" ]]; then
    echo "Project not found: $PROJ"; exit 2
  fi
  if [[ ! -d "$DATA" ]]; then
    echo "PRMI root_dir not found: $DATA"; exit 2
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "SAM2.1 checkpoint missing: $CKPT"
    echo "Run: python $PROJ/download_ckpt.py --out $CKPT"
    exit 2
  fi
  local cfg_abs
  cfg_abs="$(_locate_cfg)"
  if [[ -z "$cfg_abs" ]]; then
    echo "Could not locate sam2 config: $CFG"
    echo "Make sure sam2 is installed and the cfg path is correct.";
    exit 2
  fi
  _log "Resolved cfg: $cfg_abs"

  local json_train json_val json_test
  json_train="$(_infer_json_path_exact train)"
  json_val="$(_infer_json_path_exact val)"
  json_test="$(_infer_json_path_exact test)"
  for j in "$json_train" "$json_val" "$json_test"; do
    if [[ ! -f "$j" ]]; then
      echo "Missing JSON: $j"; exit 2
    fi
  done

  _log "Sanity-check JSON↔files (train/val/test)"
  python "$PROJ/tools/autodl_sanity_check.py" --root_dir "$DATA" --json "$json_train" --sample 64
  python "$PROJ/tools/autodl_sanity_check.py" --root_dir "$DATA" --json "$json_val" --sample 64
  python "$PROJ/tools/autodl_sanity_check.py" --root_dir "$DATA" --json "$json_test" --sample 64
  _log "Check OK"
}

_stats() {
  _log "Running PRMI stats (train+val+test)"
  mkdir -p "$PROJ/logs"
  python "$PROJ/tools/prmi_stats.py" \
    --json "$DATA/train/labels_image_gt/${SUBSET}_train.json" \
    --json "$DATA/val/labels_image_gt/${SUBSET}_val.json" \
    --json "$DATA/test/labels_image_gt/${SUBSET}_test.json" \
    --out "$PROJ/logs/stats_${SUBSET}.json"
  _log "Wrote: $PROJ/logs/stats_${SUBSET}.json"
}

_warmup() {
  _log "Warmup training (single-frame)"
  mkdir -p "$PROJ/runs_single/$SUBSET"
  local amp_flag=""
  if [[ "$AMP" == "1" ]]; then amp_flag="--amp"; fi
  python "$PROJ/train_single.py" \
    --root_dir "$DATA" \
    --json "$DATA/train/labels_image_gt/${SUBSET}_train.json" \
    --backbone sam2 \
    --sam2_cfg "$CFG" \
    --sam2_ckpt "$CKPT" \
    --sam2_image_size "$SAM2_IMAGE_SIZE" \
    --tile "$TILE" \
    --batch "$BATCH_WARMUP" \
    --epochs "$EPOCHS_WARMUP" \
    --num_workers "$NUM_WORKERS" \
    $amp_flag \
    --out_dir "$PROJ/runs_single/$SUBSET"
}

_tl() {
  _log "Time-lapse training (pair)"
  mkdir -p "$PROJ/runs_tl/$SUBSET"
  local amp_flag=""
  if [[ "$AMP" == "1" ]]; then amp_flag="--amp"; fi
  python "$PROJ/train_timelapse.py" \
    --root_dir "$DATA" \
    --json "$DATA/train/labels_image_gt/${SUBSET}_train.json" \
    --backbone sam2 \
    --sam2_cfg "$CFG" \
    --sam2_ckpt "$CKPT" \
    --sam2_image_size "$SAM2_IMAGE_SIZE" \
    --tile "$TILE" \
    --batch "$BATCH_TL" \
    --epochs "$EPOCHS_TL" \
    --num_workers "$NUM_WORKERS" \
    --tau_days "$TAU_DAYS" \
    $amp_flag \
    --out_dir "$PROJ/runs_tl/$SUBSET"
}

_infer() {
  _log "Tiled inference (single image)"
  mkdir -p "$PROJ/outputs"
  local ckpt_best="$PROJ/runs_tl/$SUBSET/checkpoints/best.pt"
  local ckpt_last="$PROJ/runs_tl/$SUBSET/checkpoints/last.pt"
  local ckpt_use=""
  if [[ -f "$ckpt_best" ]]; then ckpt_use="$ckpt_best"; elif [[ -f "$ckpt_last" ]]; then ckpt_use="$ckpt_last"; fi
  if [[ -z "$ckpt_use" ]]; then
    echo "No checkpoint found under $PROJ/runs_tl/$SUBSET/checkpoints (expected best.pt or last.pt)"; exit 2
  fi

  local json="$DATA/$SPLIT/labels_image_gt/${SUBSET}_${SPLIT}.json"
  if [[ "$SPLIT" == "val" ]]; then json="$DATA/val/labels_image_gt/${SUBSET}_val.json"; fi
  if [[ "$SPLIT" == "train" ]]; then json="$DATA/train/labels_image_gt/${SUBSET}_train.json"; fi
  if [[ "$SPLIT" == "test" ]]; then json="$DATA/test/labels_image_gt/${SUBSET}_test.json"; fi
  if [[ ! -f "$json" ]]; then echo "Missing json: $json"; exit 2; fi

  if [[ -z "$IMAGE" ]]; then
    local base
    base="$(_pick_first_image_from_json "$json")"
    IMAGE="$DATA/$SPLIT/images/$SUBSET/$base"
  fi
  if [[ ! -f "$IMAGE" ]]; then echo "Image not found: $IMAGE"; exit 2; fi

  python "$PROJ/infer_tiled.py" \
    --ckpt "$ckpt_use" \
    --image "$IMAGE" \
    --out "$PROJ/outputs/${SUBSET}_${SPLIT}_demo.png" \
    --tile "$TILE" \
    --overlap "$OVERLAP"
  _log "Wrote: $PROJ/outputs/${SUBSET}_${SPLIT}_demo.png"
}

_infer_seq() {
  _log "Sequence inference (test JSON)"
  mkdir -p "$PROJ/outputs/seq_$SUBSET"
  local ckpt_best="$PROJ/runs_tl/$SUBSET/checkpoints/best.pt"
  local ckpt_last="$PROJ/runs_tl/$SUBSET/checkpoints/last.pt"
  local ckpt_use=""
  if [[ -f "$ckpt_best" ]]; then ckpt_use="$ckpt_best"; elif [[ -f "$ckpt_last" ]]; then ckpt_use="$ckpt_last"; fi
  if [[ -z "$ckpt_use" ]]; then
    echo "No checkpoint found under $PROJ/runs_tl/$SUBSET/checkpoints (expected best.pt or last.pt)"; exit 2
  fi

  python "$PROJ/infer_sequence.py" \
    --ckpt "$ckpt_use" \
    --json "$DATA/test/labels_image_gt/${SUBSET}_test.json" \
    --root_dir "$DATA" \
    --out_dir "$PROJ/outputs/seq_$SUBSET" \
    --tile "$TILE" \
    --overlap "$OVERLAP"
}

case "$MODE" in
  check)
    _check_paths
    ;;
  stats)
    _check_paths
    _stats
    ;;
  warmup)
    _check_paths
    _warmup
    ;;
  tl)
    _check_paths
    _tl
    ;;
  infer)
    _check_paths
    _infer
    ;;
  infer_seq)
    _check_paths
    _infer_seq
    ;;
  all)
    _check_paths
    _stats
    _warmup
    _tl
    _infer
    _infer_seq
    ;;
  *)
    echo "Unknown mode: $MODE"; usage; exit 1
    ;;
esac
