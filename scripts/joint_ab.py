#!/usr/bin/env python3
"""A/B of the Phase 3 joint-score against B9 on the worst-10 sequences.

Three configs, full-length, same sequences:
  R1  joint-OFF (B9_full)            -> parity reference
  R2  joint default                  -> use_joint_score, geometric, thr 0.5, lam 0.5/0.2
  R3  joint combined (sweep-guided)  -> arithmetic, thr 0.3, lam_start 0.3, lam_min 0.0

Prints per-sequence AUC and set mean; the goal is R2/R3 >= R1 (parity or better)
with fewer knobs, per paper Table 2 (DIoU machinery worth only +0.8%).

Run with the tracking_dev env:
    /home/peoly/anaconda3/envs/tracking_dev/bin/python scripts/joint_ab.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, glob
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trackers import TrackerRegistry
from modular_evaluation import ConfigLoader

CONFIG = 'yoloe-vp-iou/ablation/B9_full.yaml'
LASOT_ROOT = '/home/peoly/datasets/lasot/test'
LIST = 'lasot_worst10_sequences.txt'
_cl = ConfigLoader()

R3 = {'use_joint_score': True, 'joint_mode': 'arithmetic',
      'reinit_joint_threshold': 0.3, 'joint_lam_start': 0.3, 'joint_lam_min': 0.0}
CONFIGS = {
    'R1_joint_off':  {},
    'R3_joint_p3':   dict(R3),                            # Phase 3 only (= B10)
    'R5_joint_p32':  dict(R3, joint_apply_phase2=True),   # Phase 3 + Phase 2 selection
}


def seq_path(name):
    return f"{LASOT_ROOT}/lasot_test_{name.rsplit('-', 1)[0]}/{name}"


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih; union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def auc_success(preds, gts):
    io = np.array([iou_xywh(p, g) if (p and g) else 0.0 for p, g in zip(preds, gts)])
    return float(np.mean([(io >= t).mean() for t in np.arange(0, 1.01, 0.01)]))


def load_list(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                out.append(line)
    return out


def run_seq(name, overrides):
    imgs = sorted(glob.glob(seq_path(name) + '/img/*.jpg'))
    gts = [[float(x) for x in l.strip().split(',')[:4]]
           for l in open(seq_path(name) + '/groundtruth.txt')][:len(imgs)]
    p = _cl.load(CONFIG); p.update(overrides); p['verbose'] = False
    t = TrackerRegistry.get_tracker(p.get('tracker', 'YOLOe-VP-IoU'), **p)
    preds = []
    for i, pth in enumerate(imgs):
        im = cv2.imread(pth)
        if i == 0:
            ok = t.initialize(im, gts[0]); preds.append(gts[0] if ok else None)
        else:
            ok, bb = t.update(im); preds.append(bb if ok else None)
    return auc_success(preds, gts)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default=LIST, help='sequence list file')
    args = ap.parse_args()
    names = list(dict.fromkeys(load_list(args.list)))   # dedup, keep order
    print(f'A/B on {len(names)} sequences ({args.list}): {", ".join(names)}\n', flush=True)

    results = {c: {} for c in CONFIGS}
    for name in names:
        row = []
        for cfg, ov in CONFIGS.items():
            try:
                a = run_seq(name, ov)
            except Exception as e:
                a = float('nan')
                print(f'  ! {name} [{cfg}] FAILED: {e}', flush=True)
            results[cfg][name] = a
            row.append(f'{cfg}={a:.3f}')
        print(f'{name:<18} ' + '  '.join(row), flush=True)

    print('\n' + '=' * 60, flush=True)
    print(f'{"config":<16}{"mean AUC":>10}{"  vs R1":>10}', flush=True)
    base = np.nanmean(list(results['R1_joint_off'].values()))
    for cfg in CONFIGS:
        m = np.nanmean(list(results[cfg].values()))
        delta = '' if cfg == 'R1_joint_off' else f'{m - base:+.4f}'
        print(f'{cfg:<16}{m:>10.4f}{delta:>10}', flush=True)


if __name__ == '__main__':
    main()
