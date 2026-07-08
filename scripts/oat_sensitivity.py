#!/usr/bin/env python3
"""One-at-a-time (OAT) hyperparameter sensitivity for YOLOe-VP-IoU.

Runs the B9_full config on a fixed sequence set, then perturbs one parameter at
a time (holding the rest at baseline) and measures the change in mean AUC. The
result ranks parameters by influence so the near-inert ones can be dropped or
frozen.

  mean AUC = mean over sequences of per-sequence success-AUC (LaSOT-style).
  |ΔAUC| below --threshold on this set => candidate for removal / fixing.

Caveats:
  * OAT does not capture interactions between parameters. For structurally
    coupled groups (adaptive triples, dual-memory weights) follow up with a
    paired sweep.
  * Cost = (1 + N_perturbations) full tracking runs over the set. Use --frames
    to cap sequence length and --params to restrict the grid while iterating.

Usage:
    python scripts/oat_sensitivity.py                       # full grid, worst-10 set
    python scripts/oat_sensitivity.py --frames 400          # cap frames for speed
    python scripts/oat_sensitivity.py --params conf iou_threshold use_kalman
    python scripts/oat_sensitivity.py --seqs volleyball-13 kite-10
    python scripts/oat_sensitivity.py --both-directions     # test up AND down
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, glob, csv, argparse, traceback
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trackers import TrackerRegistry
from modular_evaluation import ConfigLoader

CONFIG = 'yoloe-vp-iou/ablation/B9_full.yaml'
LASOT_ROOT = '/home/peoly/datasets/lasot/test'
DEFAULT_LIST = 'lasot_worst10_sequences.txt'
_cl = ConfigLoader()

# Perturbation grid: param -> list of alternative values to test (baseline excluded
# automatically). Grouped by config section; comments note the hypothesis.
PARAM_GRID = {
    # --- adaptive confidence (small range => suspected low impact) ---
    'conf':                 [0.02, 0.10],
    'conf_max':             [0.05, 0.2],
    'conf_adaptive_rate':   [10, 60],
    'iou_threshold':        [0.1, 0.3],
    # --- VPE collection ---
    'vpe_step':             [5, 20],
    'max_vpe':              [30, 80],
    'vpe_conf_threshold':   [0.02, 0.1],
    'vpe_conf_max':         [0.1, 0.4],
    'adaptive_rate':        [200, 1000],
    'max_lost_frames':      [10, 25],
    # --- Phase 2 ---
    'phase2_diou_threshold':            [-0.3, 0.0],
    'phase2_switch_validation_frames':  [0, 3],
    # --- Phase 3 ---
    'reinit_conf_threshold':                [0.2, 0.4],
    'reinit_diou_threshold':                [-0.6, -0.2],
    'reinit_diou_max':                      [-0.99, -0.7],
    'reinit_adaptive_rate':                 [50, 200],
    'phase3_redetection_validation_frames': [0, 3],
    'phase3_vpe_freeze_frames':             [0, 60],
    # --- Phase 1 high-conf re-ID ---
    'phase1_high_conf_reid_threshold':          [0.8, 0.95],
    'phase1_high_conf_reid_diou':               [-0.5, -0.1],
    'phase1_high_conf_reid_validation_frames':  [1, 5],
    'phase1_high_conf_reid_conf_gap':           [0.05, 0.3],
    # --- Kalman (suspected low impact) ---
    'use_kalman':               [False],
    'kalman_process_noise':     [0.0001, 0.01],
    'kalman_measurement_noise': [0.00001, 0.001],
    # --- dual-memory VPE ---
    'dual_long_term_capacity':        [10, 40],
    'dual_long_term_quality_threshold': [0.4, 0.8],
    'dual_long_term_weight':          [0.3, 0.7],
    'dual_short_term_weight':         [0.3, 0.7],
    'dual_temporal_decay':            [0.8, 0.95],
    'dual_update_long_term_every':    [25, 100],
    'dual_anchor_weight':             [0.0, 0.3],
    'dual_use_anchor':                [False],
    # --- experimental block (expected inert under B9) ---
    'phase3_appearance_weight': [0.0, 1.0],
    'vpe_gate_threshold':       [0.5, 0.90],
    'vpe_gate_ref':             ['anchor'],
    # --- Phase 3 joint DIoU×conf score (see joint-score-phase3) ---
    # use_joint_score tests on/off vs B9. The sub-params below are INERT unless
    # the baseline already has joint on -> run with --joint-baseline to sweep them.
    'use_joint_score':          [True],
    'joint_mode':               ['arithmetic'],   # baseline geometric; needs --joint-baseline
    'reinit_joint_threshold':   [0.3, 0.7],       # needs --joint-baseline
    'joint_lam_start':          [0.3, 0.7],       # needs --joint-baseline
    'joint_lam_min':            [0.0, 0.4],       # needs --joint-baseline
}

# Params that only bite when joint-score is the baseline (skipped in plain OAT).
JOINT_SUBPARAMS = {'joint_mode', 'reinit_joint_threshold',
                   'joint_lam_start', 'joint_lam_min'}


def seq_path(name):
    cls = name.rsplit('-', 1)[0]
    return f'{LASOT_ROOT}/lasot_test_{cls}/{name}'


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def auc_success(preds, gts):
    ious = np.array([iou_xywh(p, g) if (p and g) else 0.0 for p, g in zip(preds, gts)])
    return float(np.mean([(ious >= t).mean() for t in np.arange(0, 1.01, 0.01)]))


def load_seq_list(path):
    names = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()   # drop comments / inline notes
            if line:
                names.append(line)
    return names


def run_seq(name, overrides, num_frames):
    """Single sequence -> success-AUC, with `overrides` applied on top of B9."""
    imgs = sorted(glob.glob(seq_path(name) + '/img/*.jpg'))
    if num_frames:
        imgs = imgs[:num_frames]
    gts = []
    with open(seq_path(name) + '/groundtruth.txt') as f:
        for line in f:
            parts = line.strip().split(',')
            gts.append([float(v) for v in parts[:4]] if len(parts) >= 4 else None)
    gts = gts[:len(imgs)]

    p = _cl.load(CONFIG)
    p.update(overrides)
    p['verbose'] = False
    t = TrackerRegistry.get_tracker(p.get('tracker', 'YOLOe-VP-IoU'), **p)

    preds = []
    for idx, pth in enumerate(imgs):
        img = cv2.imread(pth)
        if idx == 0:
            ok = t.initialize(img, gts[0]); preds.append(gts[0] if ok else None)
        else:
            ok, bb = t.update(img); preds.append(bb if ok else None)
    return auc_success(preds, gts)


def run_set(names, overrides, num_frames, label):
    """Mean AUC over the sequence set; robust to per-sequence failures."""
    aucs = []
    for name in names:
        try:
            a = run_seq(name, overrides, num_frames)
            aucs.append(a)
        except Exception as e:
            print(f'    ! {name}: FAILED ({e.__class__.__name__}: {e})')
            traceback.print_exc(limit=1)
    mean = float(np.mean(aucs)) if aucs else float('nan')
    print(f'  [{label}] mean AUC = {mean:.4f}  (n={len(aucs)}/{len(names)})')
    return mean, aucs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', default=DEFAULT_LIST, help='sequence list file')
    ap.add_argument('--seqs', nargs='+', help='override list with explicit sequences')
    ap.add_argument('--params', nargs='+', help='restrict grid to these params')
    ap.add_argument('--frames', type=int, default=0, help='cap frames/seq (0=all)')
    ap.add_argument('--both-directions', action='store_true',
                    help='report every perturbation (default already tests all grid values)')
    ap.add_argument('--joint-baseline', action='store_true',
                    help='run baseline with use_joint_score=True so joint sub-params become active')
    ap.add_argument('--threshold', type=float, default=0.005,
                    help='|ΔAUC| below this => flagged low-impact')
    ap.add_argument('--out', default='oat_sensitivity.csv', help='output CSV')
    args = ap.parse_args()

    names = args.seqs if args.seqs else load_seq_list(args.list)

    # Baseline overrides applied to every run (baseline + perturbations).
    base_overrides = {'use_joint_score': True} if args.joint_baseline else {}

    grid = dict(PARAM_GRID)
    if args.joint_baseline:
        grid.pop('use_joint_score', None)        # already the baseline
    else:
        for k in JOINT_SUBPARAMS:                 # inert without a joint-on baseline
            grid.pop(k, None)
    if args.params:
        missing = [p for p in args.params if p not in PARAM_GRID]
        if missing:
            sys.exit(f'Unknown params: {missing}')
        grid = {k: PARAM_GRID[k] for k in args.params}

    base_params = _cl.load(CONFIG)
    base_params.update(base_overrides)
    print(f'Sequences ({len(names)}): {", ".join(names)}')
    print(f'Params: {len(grid)} | frames/seq: {args.frames or "all"}\n')

    label = 'B9_full' + (' + joint-score' if args.joint_baseline else '')
    print(f'=== BASELINE ({label}) ===')
    base_mean, _ = run_set(names, dict(base_overrides), args.frames, 'baseline')

    rows = []
    print('\n=== PERTURBATIONS ===')
    for param, values in grid.items():
        baseval = base_params.get(param, '<unset>')
        for v in values:
            if v == baseval:
                continue
            print(f'\n{param}: {baseval} -> {v}')
            overrides = dict(base_overrides); overrides[param] = v
            mean, _ = run_set(names, overrides, args.frames, f'{param}={v}')
            delta = mean - base_mean
            rows.append(dict(param=param, baseline_value=baseval, test_value=v,
                             mean_auc=round(mean, 4), delta_auc=round(delta, 4),
                             abs_delta=round(abs(delta), 4)))

    # Rank each param by its worst-case |ΔAUC| across the tested values
    per_param = {}
    for r in rows:
        per_param.setdefault(r['param'], []).append(r)
    ranking = sorted(per_param.items(),
                     key=lambda kv: max(x['abs_delta'] for x in kv[1]), reverse=True)

    print('\n' + '=' * 68)
    print(f'BASELINE mean AUC = {base_mean:.4f}   (set: {len(names)} seqs)')
    print('Parameters ranked by max |ΔAUC| (most influential first):')
    print(f'{"param":<38}{"max|Δ|":>8}{"  perturbations (value:Δ)"}')
    print('-' * 68)
    low_impact = []
    for param, rs in ranking:
        maxabs = max(x['abs_delta'] for x in rs)
        detail = ', '.join(f'{x["test_value"]}:{x["delta_auc"]:+.3f}' for x in rs)
        flag = '  <-- LOW IMPACT' if maxabs < args.threshold else ''
        if maxabs < args.threshold:
            low_impact.append(param)
        print(f'{param:<38}{maxabs:>8.3f}  {detail}{flag}')

    print('-' * 68)
    print(f'{len(low_impact)}/{len(per_param)} params below |ΔAUC|={args.threshold} '
          f'-> prune/fix candidates:')
    print('  ' + (', '.join(low_impact) if low_impact else '(none)'))

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['param', 'baseline_value', 'test_value',
                                          'mean_auc', 'delta_auc', 'abs_delta'])
        w.writeheader(); w.writerows(rows)
    print(f'\nWrote {args.out} ({len(rows)} rows).')


if __name__ == '__main__':
    main()
