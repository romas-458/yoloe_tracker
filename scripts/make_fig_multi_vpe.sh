#!/usr/bin/env bash
# ============================================================================
# Побудова фігури "як працює мульти-VPE" (figures/fig4_multi_vpe.png).
#
# Проганяє одне GOT-10k відео двічі — single-VPE (D0) і multi-VPE (D5, verbose) —
# з покадровим дампом IoU, потім будує порівняльну фігуру:
#   верх  — дві траси IoU (single vs multi)
#   низ   — розходження видів пам'яті (anchor/LT/ST) + число активних видів K
#
# Використання:
#   bash scripts/make_fig_multi_vpe.sh [VIDEO] [OUT_PNG]
#     VIDEO   — назва GOT-10k послідовності (типово GOT-10k_Val_000111)
#     OUT_PNG — шлях фігури (типово figures/fig4_multi_vpe.png)
#
# Приклади сильного виграшу multi: GOT-10k_Val_000111, _000028, _000025.
# ============================================================================
set -euo pipefail

VIDEO="${1:-GOT-10k_Val_000111}"
OUT_PNG="${2:-figures/fig4_multi_vpe.png}"

# приймаємо і назву, і повний шлях до теки послідовності
VIDEO="$(basename "$VIDEO")"

PYTHON="${PYTHON:-/home/peoly/anaconda3/envs/tracking_dev/bin/python}"
DATA="${DATA:-/home/peoly/datasets/got10k/got_10k_val/val}"
CFG_DIR="yoloe-vp-iou/ablation"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "▶ відео: $VIDEO"
echo "▶ робоча тека: $WORK"

# --- 1. single-VPE (D0) ------------------------------------------------------
echo "▶ [1/3] single-VPE (D0_minimal)…"
"$PYTHON" scripts/modular_evaluation.py -d "$DATA" \
    -o "$WORK/single" --tracker-config "$CFG_DIR/D0_minimal.yaml" \
    --dataset got10k --repetitions 1 --video "$VIDEO" --dump-per-frame \
    --num-frames 100000 > "$WORK/single.log" 2>&1

# --- 2. multi-VPE (D5, verbose) ---------------------------------------------
echo "▶ [2/3] multi-VPE (D5_multi_vpe, verbose)…"
sed 's/^verbose: false/verbose: true/' \
    "configs/$CFG_DIR/D5_multi_vpe.yaml" > "$WORK/D5_verbose.yaml"
"$PYTHON" scripts/modular_evaluation.py -d "$DATA" \
    -o "$WORK/multi" --tracker-config "$WORK/D5_verbose.yaml" \
    --dataset got10k --repetitions 1 --video "$VIDEO" --dump-per-frame \
    --num-frames 100000 2>&1 \
  | tr '\r' '\n' | grep '^MVPE_VIEWS' > "$WORK/views.txt" || true

n_views=$(wc -l < "$WORK/views.txt")
echo "  MVPE_VIEWS: $n_views точок оновлення пам'яті"
if [ "$n_views" -eq 0 ]; then
    echo "  ⚠️  жодного MVPE_VIEWS — multi-VPE не активувався (перевір multi_vpe_prompt / dual memory)"
fi

# --- 3. фігура ---------------------------------------------------------------
echo "▶ [3/3] побудова фігури…"
mkdir -p "$(dirname "$OUT_PNG")"
python3 scripts/fig_multi_vpe.py "$VIDEO" \
    --single-npz "$WORK/single/curves_YOLOe-VP-IoU.npz" \
    --multi-npz  "$WORK/multi/curves_YOLOe-VP-IoU.npz" \
    --views      "$WORK/views.txt" \
    --out "$OUT_PNG"

echo "✔ готово: $OUT_PNG"
