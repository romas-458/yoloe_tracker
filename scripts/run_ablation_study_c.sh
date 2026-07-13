#!/usr/bin/env bash
# ============================================================================
# C-series ablation: leave-one-out від C0_full на збалансованому наборі з 12 відео
# ============================================================================
# На відміну від B-серії (кумулятивна: B1→B9), тут кожен конфіг вимикає рівно
# один компонент при всіх інших увімкнених. Внесок компонента = C0 − Cn.
#
# Набір: lasot_ablation_subset12.txt (НЕ десятка з Таблиці 2 — вона зміщена
# на +7 AUC вгору і не містить жодного відео з Fast Motion).
#
# ⚠️ SE(Δ) на 12 відео ≈ 0.037. Різниці менші за ~0.04 AUC не значущі.
#    Фінальні числа для статті — з повного прогону 280 відео найкращого конфігу.
#
# Використання:  bash scripts/run_ablation_study_c.sh [конфіг ...]
#   без аргументів — прогнати всі C0..C12
# ============================================================================
set -u

# YOLOE живе лише в conda-env tracking_dev; base його не має
PYTHON="${PYTHON:-/home/peoly/anaconda3/envs/tracking_dev/bin/python}"

DATASET_PATH="/home/peoly/datasets/lasot/test"
TEST_LIST="lasot_ablation_subset12.txt"
OUTPUT_DIR="results_ablation_c"
NUM_FRAMES=100000

CONFIGS=(
    C0_full            # еталон
    C1_no_joint        # ⭐ головна гіпотеза: joint-score — чистий мінус
    C2_no_vpe
    C3_no_kalman
    C4_no_dual_memory
    C5_no_p1_reid      # правильне вимкнення (1.0), не 0.0 як у B8
    C6_no_diou_p2
    C7_no_adaptive
    C8_no_vpe_gate
    C9_no_p3_appearance
    C10_model_v8l
    C11_floor095
    C12_floor070
)
[ "$#" -gt 0 ] && CONFIGS=("$@")

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "YOLOe-VP-IoU Ablation Study (C-series, LOO)"
echo "Dataset:   $DATASET_PATH"
echo "Test list: $TEST_LIST ($(grep -cvE '^\s*#|^\s*$' "$TEST_LIST") відео)"
echo "Output:    $OUTPUT_DIR"
echo "========================================="

for exp in "${CONFIGS[@]}"; do
    config="yoloe-vp-iou/ablation/${exp}.yaml"
    out="$OUTPUT_DIR/${exp}"

    if [ -f "$out/summary_YOLOe-VP-IoU.json" ]; then
        echo "--- $exp: вже є результат, пропускаю (видаліть $out щоб перезапустити)"
        continue
    fi

    echo ""
    echo "--- Running $exp"
    "$PYTHON" scripts/modular_evaluation.py \
        -d "$DATASET_PATH" \
        -o "$out" \
        --tracker-config "$config" \
        --dataset lasot \
        --test-list "$TEST_LIST" \
        --num-frames "$NUM_FRAMES" \
        2>&1 | tee "$OUTPUT_DIR/${exp}.log" | grep -E "Відео:|AUC|FPS|ERROR|Traceback" || true
done

echo ""
echo "========================================="
echo "Зведення (внесок компонента = C0 − Cn):"
"$PYTHON" - <<'PY'
import json, os, glob
D='results_ablation_c'
rows={}
for f in glob.glob(f'{D}/*/summary_YOLOe-VP-IoU.json'):
    exp=os.path.basename(os.path.dirname(f))
    o=json.load(open(f))['overall']
    rows[exp]=(o['avg_auc'], o['avg_fps'])
if 'C0_full' not in rows:
    print('C0_full ще не прогнано — внески не рахую'); raise SystemExit
base=rows['C0_full'][0]
print(f'{"конфіг":22}{"AUC":>8}{"FPS":>8}{"внесок":>9}  значущість')
def key(e):
    tag=e.split('_')[0][1:]                       # "0", "1", "1b", "12"
    num=int(''.join(c for c in tag if c.isdigit()))
    suf=''.join(c for c in tag if not c.isdigit())  # "" сортується перед "b"
    return (num, suf)
for e in sorted(rows, key=key):
    auc,fps=rows[e]
    d=base-auc
    sig='' if e=='C0_full' else ('значущий' if abs(d)>=0.04 else 'у межах шуму')
    print(f'{e:22}{auc:8.4f}{fps:8.1f}{"" if e=="C0_full" else f"{d:+9.4f}"}  {sig}')
PY
