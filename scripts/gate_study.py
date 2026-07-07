#!/usr/bin/env python3
"""VPE-gate study for the YOLOe-VP-IoU tracker.

Investigates the appearance gate in `_collect_vpe`:
  * closeness (scaled cosine sim) of each collected VPE to the gate reference,
    for both `vpe_gate_ref` modes ('aggregated' vs 'anchor');
  * how the gate threshold maps to accepted/rejected instances and to tracking
    AUC (real runs, so the cascade/feedback effect is captured);
  * which frames get rejected (object crops at sub-threshold frames).

Runs the B9_full config on chosen LaSOT sequences and writes PNG figures.

Usage:
    python scripts/gate_study.py                 # basketball-1 + tiger-4
    python scripts/gate_study.py --seqs airplane-1 drone-15 --frames 600
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
# gate off (tiny) gives unperturbed closeness + baseline AUC; then real thresholds
THRESHOLDS = [1e-9, 0.75, 0.80, 0.85, 0.90]
REFS = ['aggregated', 'anchor']
CROP_REF, CROP_THR = 'aggregated', 0.85     # run used for rejection-frame crops
_cl = ConfigLoader()


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


def build(threshold, ref):
    p = _cl.load(CONFIG)
    p['vpe_gate_threshold'] = threshold
    p['vpe_gate_ref'] = ref
    return TrackerRegistry.get_tracker(p.get('tracker', 'YOLOe-VP-IoU'), **p)


def run(name, threshold, ref, keep_crops=False):
    imgs = sorted(glob.glob(seq_path(name) + '/img/*.jpg'))[:NUM_FRAMES]
    gts = []
    with open(seq_path(name) + '/groundtruth.txt') as f:
        for line in f:
            parts = line.strip().split(',')
            gts.append([float(v) for v in parts[:4]] if len(parts) >= 4 else None)
    gts = gts[:len(imgs)]

    t = build(threshold, ref)
    sims, crops = [], {}
    t._in_collect = False
    orig_collect, orig_sim = t._collect_vpe, t._calculate_cosine_similarity

    def collect_wrap(image, bbox, *a, **k):
        t._in_collect = True
        try:
            if keep_crops:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                pad = 6
                crop = image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
                crops[t.frame_count] = crop
            return orig_collect(image, bbox, *a, **k)
        finally:
            t._in_collect = False

    def sim_wrap(v1, v2):
        r = orig_sim(v1, v2)
        if getattr(t, '_in_collect', False):   # only the gate's comparison
            sims.append((t.frame_count, r))
        return r

    t._collect_vpe, t._calculate_cosine_similarity = collect_wrap, sim_wrap

    preds = []
    for idx, pth in enumerate(imgs):
        img = cv2.imread(pth)
        if idx == 0:
            ok = t.initialize(img, gts[0]); preds.append(gts[0] if ok else None)
        else:
            ok, bb = t.update(img); preds.append(bb if ok else None)

    return dict(sims=sims, crops=crops, auc=auc_success(preds, gts),
                accepted=t.vpe_gate_accepted, rejected=t.vpe_gate_rejected)


def study(names):
    data = {}
    for name in names:
        print(f'\n===== {name} =====')
        data[name] = {}
        for ref in REFS:
            data[name][ref] = {}
            for thr in THRESHOLDS:
                keep = (ref == CROP_REF and abs(thr - CROP_THR) < 1e-6)
                r = run(name, thr, ref, keep_crops=keep)
                data[name][ref][thr] = r
                tag = 'off' if thr < 1e-6 else f'{thr:.2f}'
                print(f'  ref={ref:<10} thr={tag:<4} -> AUC={r["auc"]:.3f} '
                      f'acc={r["accepted"]:<3} rej={r["rejected"]}')
    return data


# ----------------------------- figures -----------------------------
def fig_closeness(data, names, out):
    fig, axes = plt.subplots(1, len(names), figsize=(7.5 * len(names), 5), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[0][j]
        for ref, color in [('aggregated', 'tab:blue'), ('anchor', 'tab:red')]:
            s = data[name][ref][1e-9]['sims']
            if not s:
                continue
            fr = [f for f, _ in s]; sm = [v for _, v in s]
            ax.plot(fr, sm, '-o', ms=3, lw=1, color=color, alpha=0.7,
                    label=f"ref={ref} (min={min(sm):.2f}, mean={np.mean(sm):.2f})")
        ax.axhline(0.75, color='k', ls='--', lw=1, alpha=0.6, label='B9 thr=0.75')
        ax.set_title(f'{name}: closeness of collected VPE to gate ref')
        ax.set_xlabel('frame'); ax.set_ylabel('scaled cos sim (cos+1)/2')
        ax.set_ylim(0.55, 1.005); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); print('saved', out)


def fig_auc(data, names, out):
    fig, axes = plt.subplots(1, len(names), figsize=(7.5 * len(names), 5), squeeze=False)
    thrs = [t for t in THRESHOLDS if t > 1e-6]
    for j, name in enumerate(names):
        ax = axes[0][j]; ax2 = ax.twinx()
        for ref, c in [('aggregated', 'tab:blue'), ('anchor', 'tab:red')]:
            base = data[name][ref][1e-9]['auc']
            aucs = [data[name][ref][t]['auc'] for t in thrs]
            rejs = [data[name][ref][t]['rejected'] for t in thrs]
            ax.plot(thrs, aucs, '-o', color=c, label=f'AUC ref={ref}')
            ax.axhline(base, color=c, ls=':', lw=1, alpha=0.6)
            ax2.plot(thrs, rejs, '--s', color=c, alpha=0.45, ms=4)
        ax.set_title(f'{name}: AUC vs gate threshold\n(dotted = gate off; dashed = #rejected, right axis)')
        ax.set_xlabel('vpe_gate_threshold'); ax.set_ylabel('tracking AUC')
        ax2.set_ylabel('# rejected VPE'); ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); print('saved', out)


def fig_rejections(data, name, out):
    r = data[name][CROP_REF][CROP_THR]
    sim_by_frame = dict(r['sims'])
    rej = sorted(f for f, s in r['sims'] if s < CROP_THR)
    if not rej:
        print(f'no rejections for {name} at thr={CROP_THR}; skip crop figure'); return
    rej = rej[:12]
    n = len(rej); cols = min(6, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows), squeeze=False)
    for k, fr in enumerate(rej):
        ax = axes[k // cols][k % cols]
        crop = r['crops'].get(fr)
        if crop is not None and crop.size:
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f'frame {fr}\nsim={sim_by_frame[fr]:.3f}', fontsize=9, color='tab:red')
        ax.axis('off')
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle(f'{name}: VPE rejected by gate (ref={CROP_REF}, thr={CROP_THR}) — {r["rejected"]} total',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=110); print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', default=['basketball-1', 'tiger-4'])
    ap.add_argument('--frames', type=int, default=900)
    ap.add_argument('--outdir', default='/tmp')
    args = ap.parse_args()
    NUM_FRAMES = args.frames
    data = study(args.seqs)
    fig_closeness(data, args.seqs, f'{args.outdir}/gate_closeness.png')
    fig_auc(data, args.seqs, f'{args.outdir}/gate_auc.png')
    fig_rejections(data, args.seqs[-1], f'{args.outdir}/gate_rejections.png')
