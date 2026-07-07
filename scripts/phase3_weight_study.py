#!/usr/bin/env python3
"""Impact of `phase3_appearance_weight` on the YOLOe-VP-IoU tracker.

The weight only affects Phase 3 re-detection candidate selection:
    score = (1-w)*conf + w*sim(candidate_VPE, memory_ref)
so it can only change the outcome when the object is lost AND there is
more than one eligible candidate. This script sweeps the weight, measures
tracking AUC, and — as leverage context — counts how often Phase 3 had a
multi-candidate choice and how often appearance overrode the max-conf pick.

Usage:
    python scripts/phase3_weight_study.py
    python scripts/phase3_weight_study.py --seqs basketball-1 person-1 --frames 500
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, glob, argparse
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trackers import TrackerRegistry
from modular_evaluation import ConfigLoader

CONFIG = 'yoloe-vp-iou/ablation/B9_full.yaml'
LASOT_ROOT = '/home/peoly/datasets/lasot/test'
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
_cl = ConfigLoader()


def seq_path(name):
    return f'{LASOT_ROOT}/lasot_test_{name.rsplit("-", 1)[0]}/{name}'


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


def run(name, weight):
    imgs = sorted(glob.glob(seq_path(name) + '/img/*.jpg'))[:NUM_FRAMES]
    gts = []
    with open(seq_path(name) + '/groundtruth.txt') as f:
        for line in f:
            p = line.strip().split(',')
            gts.append([float(v) for v in p[:4]] if len(p) >= 4 else None)
    gts = gts[:len(imgs)]

    p = _cl.load(CONFIG)
    p['phase3_appearance_weight'] = weight
    t = TrackerRegistry.get_tracker(p.get('tracker', 'YOLOe-VP-IoU'), **p)

    stats = {'multi': 0, 'diverged': 0}
    orig_sel = t._select_phase3_candidate

    def sel_wrap(image, eligible, tag=""):
        res = orig_sel(image, eligible, tag)
        if len(eligible) > 1:
            stats['multi'] += 1
            if res is not max(eligible, key=lambda c: c['conf']):
                stats['diverged'] += 1
        return res
    t._select_phase3_candidate = sel_wrap

    preds = []
    for idx, pth in enumerate(imgs):
        img = cv2.imread(pth)
        if idx == 0:
            ok = t.initialize(img, gts[0]); preds.append(gts[0] if ok else None)
        else:
            ok, bb = t.update(img); preds.append(bb if ok else None)
    return dict(auc=auc_success(preds, gts), **stats)


def study(names):
    data = {}
    for name in names:
        print(f'\n===== {name} =====')
        data[name] = {}
        for w in WEIGHTS:
            r = run(name, w)
            data[name][w] = r
            print(f'  w={w:.2f} -> AUC={r["auc"]:.3f}  '
                  f'phase3_multi_candidate={r["multi"]:<3} appearance_overrode_conf={r["diverged"]}')
    return data


def figure(data, names, out):
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.6), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[0][j]
        aucs = [data[name][w]['auc'] for w in WEIGHTS]
        base = data[name][0.0]['auc']
        ax.plot(WEIGHTS, aucs, '-o', color='tab:blue', label='AUC')
        ax.axhline(base, color='gray', ls=':', lw=1, label=f'w=0 baseline ({base:.3f})')
        mx = max(WEIGHTS, key=lambda w: data[name][w]['auc'])
        ax.plot([mx], [data[name][mx]['auc']], 'r*', ms=14, label=f'best w={mx}')
        multi = data[name][WEIGHTS[-1]]['multi']
        ax.set_title(f'{name}\n(Phase3 multi-candidate events: {multi})')
        ax.set_xlabel('phase3_appearance_weight'); ax.set_ylabel('tracking AUC')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('phase3_appearance_weight sweep — B9_full config', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=110)
    print('\nsaved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+',
                    default=['basketball-1', 'person-1', 'drone-15', 'tiger-4'])
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--outdir', default='/tmp')
    args = ap.parse_args()
    NUM_FRAMES = args.frames
    data = study(args.seqs)
    figure(data, args.seqs, f'{args.outdir}/phase3_weight.png')
