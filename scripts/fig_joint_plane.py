#!/usr/bin/env python3
"""
Фігура: площина (DIoU, conf) з ізолініями joint-балу та anti-teleport floor.

Показує механізм, а не результат. Кожна точка — детекція одного кадру Phase 3.
Ізолінії — рівні joint-балу при тому λ, яке діяло на цьому кадрі.
Дві вертикалі — reinit_diou_floor -0.80 (що був) і -0.70 (що став).

Дані: рядки JOINTCAND з verbose-прогону трекера (див. yoloe_vp_iou_tracker.py).
Справжня ціль визначається за IoU з groundtruth того ж кадру.

Використання:
    python scripts/fig_joint_plane.py <cands.txt> <video> [--frame N] [--out path]
Без --frame обирається кадр, де є ≥2 кандидати і найкращий за балом НЕ є ціллю
(тобто кадр перехоплення дистрактором).
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED = '#2a78d6', '#e34948'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.labelcolor': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
})

FIELD = re.compile(r'(\w+)=([^\s]+)')


def parse(path):
    rows = []
    for line in open(path):
        if not line.startswith('JOINTCAND'):
            continue
        d = dict(FIELD.findall(line))
        rows.append(dict(
            frame=int(d['frame']), lam=float(d['lam']), floor=float(d['floor']),
            thr=float(d['thr']), mode=d['mode'], conf=float(d['conf']),
            diou=float(d['diou']), score=float(d['score']),
            box=[float(d['x1']), float(d['y1']), float(d['x2']), float(d['y2'])],
            status=d['status']))
    return rows


def load_gt(video):
    cat = video.rsplit('-', 1)[0]
    p = Path(f'/home/peoly/datasets/lasot/test/lasot_test_{cat}/{video}/groundtruth.txt')
    gt = []
    for line in open(p):
        x, y, w, h = (float(v) for v in line.strip().replace('\t', ',').split(','))
        gt.append([x, y, x + w, y + h])
    return gt


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def joint(conf, diou, lam, mode):
    dn = (diou + 1.0) / 2.0
    if mode == 'arithmetic':
        return (1 - lam) * conf + lam * dn
    return np.power(np.maximum(conf, 1e-6), 1 - lam) * np.power(np.maximum(dn, 1e-6), lam)


def pick_frame(rows, gt):
    """кадр, де найкращий за joint-балом кандидат — НЕ ціль (перехоплення)"""
    by = {}
    for r in rows:
        by.setdefault(r['frame'], []).append(r)
    best = None
    for f, cands in sorted(by.items()):
        if len(cands) < 2 or f >= len(gt):
            continue
        elig = [c for c in cands if c['status'] == 'eligible']
        if not elig:
            continue
        top = max(elig, key=lambda c: c['score'])
        ious = [iou(c['box'], gt[f]) for c in cands]
        if max(ious) < 0.3:            # цілі взагалі немає серед детекцій
            continue
        true_i = int(np.argmax(ious))
        if cands[true_i] is not top and top['score'] >= top['thr']:
            margin = top['score'] - joint(cands[true_i]['conf'], cands[true_i]['diou'],
                                          top['lam'], top['mode'])
            if best is None or margin < best[1]:   # найтісніше перехоплення = найнаочніше
                best = (f, margin)
    return best[0] if best else None


def make(rows, gt, frame, out, video):
    cands = [r for r in rows if r['frame'] == frame]
    lam, mode, thr = cands[0]['lam'], cands[0]['mode'], cands[0]['thr']
    ious = [iou(c['box'], gt[frame]) for c in cands]
    true_i = int(np.argmax(ious))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    # ізолінії joint-балу
    dd, cc = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(0, 1, 400))
    Z = joint(cc, dd, lam, mode)
    ax.contour(dd, cc, Z, levels=np.arange(0.1, 1.0, 0.1),
               colors=GRID, linewidths=0.7, zorder=1)
    cs = ax.contour(dd, cc, Z, levels=[thr], colors=[INK2], linewidths=1.6, zorder=2)
    ax.clabel(cs, fmt=lambda v: f'поріг прийняття {v:g}', fontsize=7.6, colors=INK2)

    # зони відсічки: те, що різав старий floor, і смуга, яку додатково ріже новий
    ax.axvspan(-1, -0.8, color=RED, alpha=0.05, zorder=0)
    ax.axvspan(-0.8, -0.7, color=RED, alpha=0.14, zorder=0)
    ax.axvline(-0.8, color=RED, linewidth=1.4, linestyle='--', zorder=3)
    ax.axvline(-0.7, color=RED, linewidth=1.8, linestyle='-', zorder=3)
    ax.annotate('floor −0.80', (-0.8, 0.985), ha='right', va='top', fontsize=7.6,
                color=RED, xytext=(-4, 0), textcoords='offset points')
    ax.annotate('floor −0.70', (-0.7, 0.985), ha='left', va='top', fontsize=7.6,
                color=RED, xytext=(4, 0), textcoords='offset points')
    ax.annotate('смуга, яку додатково\nріже floor −0.70', (-0.74, 0.72),
                xytext=(0.10, 0.82), ha='left', va='center', fontsize=7.6, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, linewidth=0.8))

    offsets = {}   # рознесені підписи, щоб не налазили
    for i, c in enumerate(cands):
        is_true = (i == true_i)
        ax.scatter(c['diou'], c['conf'], s=150 if is_true else 80,
                   marker='*' if is_true else 'o',
                   c=BLUE if is_true else RED, edgecolors=SURFACE, linewidths=0.9, zorder=5)
        if is_true:
            tag, xy = 'ЦІЛЬ', (16, -26)
        elif c['status'] == 'below_floor':
            tag, xy = 'відкинутий обома floor', (16, -2)
        else:
            tag, xy = 'ДИСТРАКТОР — перемагає', (16, 12)
        lbl = (f"{tag}\nconf {c['conf']:.3f} · DIoU {c['diou']:.3f} · бал {c['score']:.3f}"
               if c['status'] == 'eligible' else f"{tag}\nбал {c['score']:.3f}")
        ax.annotate(lbl,
                    (c['diou'], c['conf']), textcoords='offset points',
                    xytext=xy, fontsize=7.6, color=INK2,
                    arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.6,
                                    shrinkA=0, shrinkB=6))

    top = max((c for c in cands if c['status'] == 'eligible'), key=lambda c: c['score'])
    hijack = cands[true_i] is not top

    ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('DIoU з опорною позицією  (далеко ← → близько)')
    ax.set_ylabel('впевненість детектора (conf)')
    # запас цілі над новою відсічкою — вузький, і це варто показати чесно
    margin = cands[true_i]['diou'] - (-0.7)
    ax.annotate(f'запас цілі над новою\nвідсічкою — лише {margin:.3f} DIoU',
                (cands[true_i]['diou'], cands[true_i]['conf'] - 0.02),
                xytext=(0.10, 0.50), ha='left', va='center', fontsize=7.4, color=INK2,
                arrowprops=dict(arrowstyle='->', color=MUTED, linewidth=0.7))

    ttl = ('Дистрактор виграє з гіршою близькістю — за рахунок вищої впевненості'
           if hijack else 'Кандидати Phase 3 у площині (DIoU, conf)')
    ax.set_title(f'{ttl}\n{video}, кадр {frame} · λ={lam:.2f} · '
                 f'бал = {(1 - lam):.1f}·conf + {lam:.1f}·(DIoU+1)/2',
                 loc='left', color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print('  ✓', out, f'(кадр {frame}, {len(cands)} кандидатів, hijack={hijack})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cands'); ap.add_argument('video')
    ap.add_argument('--frame', type=int); ap.add_argument('--out', default='figures/fig3_joint_plane.png')
    a = ap.parse_args()
    rows = parse(a.cands)
    gt = load_gt(a.video)
    f = a.frame if a.frame is not None else pick_frame(rows, gt)
    if f is None:
        raise SystemExit('кадру з перехопленням не знайдено — задайте --frame вручну')
    make(rows, gt, f, a.out, a.video)
