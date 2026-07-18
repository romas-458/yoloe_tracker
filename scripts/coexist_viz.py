#!/usr/bin/env python3
"""
Диспетчерська MOT-виключення за співіснуванням: уся картина одним кадром на відео.

Ганяє СПРАВЖНІЙ трекер із use_coexist_exclusion=True (D14) і кожен кадр читає його
живий пул tracklet-ів (tracker.coexist.tracks). Показує, ЧОМУ вето рятує book-19 і
чому отруює kite-10 — в одній фігурі.

Панелі (спільна вісь X = кадр):
  1. Доріжки tracklet-ів (swimlane): кожен об'єкт кадру — своя горизонтальна смуга на
     весь час життя. Колір: сірий норм / бурштин поки копиться доказ роз'єднаності /
     ⊗ у момент штампа + червона рамка після. Крізь доріжки йдуть дві траєкторії:
       • зелена — де ЦІЛЬ насправді (tracklet із max IoU до GT);
       • синя  — що ТРИМАЄ трекер. Стрибок синьої на іншу доріжку = хибний лок.
  2. IoU до GT у часі: D14 vs D0 (наслідок — де виграш/регресія).
  3. Стрічка подій вето: зелений тік = заблокували дистрактора (користь),
     червоний = заблокували справжню ціль (податок/отруєння).

Прогонити конда-оточенням tracking_dev:
  /home/peoly/anaconda3/envs/tracking_dev/bin/python scripts/coexist_viz.py \
      --video book-19 --out figures/coexist/book-19.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from trackers import TrackerRegistry            # noqa: E402
from config_loader import ConfigLoader          # noqa: E402
from vpe_drift_study import (find_video, load_gt, iou, load_flags,  # noqa: E402
                             BLUE, RED, AQUA, PLUM, INK, INK2, MUTED,
                             GRID, AXIS, SURFACE)

AMBER = '#d98a1f'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 11,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.labelcolor': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def xywh_to_xyxy(b):
    return None if b is None else [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def run_tracker(video_path, images, gts, config, model, imgsz, stamp_min=None,
                capture_pool=False):
    """Один прогін. Повертає per-frame ious + (якщо capture_pool) знімки пулу.

    Знімок кадру: {tid: dict(box, stamped, disjoint, matched)} для живих tracklet-ів,
    + gt_tid / held_tid (доріжки, що володіють GT-боксом і боксом трекера).
    """
    params = {'model_path': model, 'imgsz': imgsz}
    params.update(ConfigLoader().load(config))
    if stamp_min is not None:
        params['coexist_stamp_min_frames'] = stamp_min
    tracker = TrackerRegistry.get_tracker('YOLOe-VP-IoU', **params)

    ious, snaps = [], []
    for idx, img_path in enumerate(tqdm(images, desc=Path(config).stem, leave=False)):
        image = cv2.imread(str(img_path))
        if image is None:
            ious.append(np.nan)
            snaps.append(None)
            continue
        gt = gts[idx]
        if idx == 0:
            ok = tracker.initialize(image, gt)
            pred = gt if ok else None
        else:
            ok, pred = tracker.update(image)
            if not ok:
                pred = None
        ious.append(iou(pred, gt))

        if not capture_pool:
            continue
        pool = getattr(tracker, 'coexist', None)
        if pool is None:
            snaps.append(None)
            continue
        gt_xyxy, pred_xyxy = xywh_to_xyxy(gt), xywh_to_xyxy(pred)
        live = {}
        gt_tid, gt_best = None, 0.3
        held_tid, held_best = None, 0.3
        for t in pool.tracks:
            live[t.id] = dict(box=list(t.box), stamped=t.stamped,
                              disjoint=t.disjoint, matched=(t.misses == 0))
            if t.misses == 0:                       # присутній цього кадру
                g = iou_xyxy(t.box, gt_xyxy)
                if g > gt_best:
                    gt_best, gt_tid = g, t.id
                h = iou_xyxy(t.box, pred_xyxy)
                if h > held_best:
                    held_best, held_tid = h, t.id
        snaps.append(dict(live=live, gt_tid=gt_tid, held_tid=held_tid))
    return ious, snaps


def build(video, snaps, iou14, iou0, absent, title, out, min_life=25):
    n = len(snaps)
    frames = np.arange(n)

    # життєві межі + подієві дані по кожному tracklet-у
    seen = {}                       # tid -> [frames present]
    stamp_at = {}                   # tid -> перший кадр штампа
    disjoint_frac = {}              # (tid, frame) -> disjoint/30 для кольору
    for f, s in enumerate(snaps):
        if s is None:
            continue
        for tid, d in s['live'].items():
            if not d['matched']:
                continue
            seen.setdefault(tid, []).append(f)
            disjoint_frac[(tid, f)] = min(1.0, d['disjoint'] / 30.0)
            if d['stamped'] and tid not in stamp_at:
                stamp_at[tid] = f

    # Ключові доріжки — повні swimlanes. Решта (ефемерні уламки IoU-асоціації, яких на
    # дрібних швидких об'єктах — сотні) згортаються в один рядок-щільність: інакше
    # фрагментація цілі робить фігуру нечитною. Сам факт фрагментації видно у тому
    # рядку — це і є причина отруєння (ціль без стабільного id не звільнити від штампа).
    gt_lanes = {s['gt_tid'] for s in snaps if s and s['gt_tid'] is not None}
    held_lanes = {s['held_tid'] for s in snaps if s and s['held_tid'] is not None}
    key = gt_lanes | held_lanes | set(stamp_at) | {
        t for t in seen if len(seen[t]) >= min_life}
    ephemeral = [t for t in seen if t not in key]
    order = sorted(key, key=lambda t: seen[t][0])
    lane_y = {tid: i for i, tid in enumerate(order)}
    H = len(order) + (1 if ephemeral else 0)
    eph_y = len(order)              # рядок згорнутих уламків унизу

    fig = plt.figure(figsize=(15, 4.2 + 0.16 * H), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.16 * H + 1.6, 1.5, 0.55], hspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ── Панель 1: swimlanes ────────────────────────────────────────────────
    for tid in order:
        y = lane_y[tid]
        fs = np.array(seen[tid])
        # суцільні відрізки життя
        for a, b in _runs(fs):
            ax1.add_patch(Rectangle((a, y - 0.32), b - a + 1, 0.64,
                                    facecolor='#e9e8e1', edgecolor='none', zorder=1))
        # бурштин поки копиться доказ роз'єднаності
        for f in fs:
            fr = disjoint_frac.get((tid, f), 0.0)
            if 0 < fr < 1.0:
                ax1.add_patch(Rectangle((f, y - 0.32), 1, 0.64,
                              facecolor=AMBER, alpha=0.15 + 0.55 * fr,
                              edgecolor='none', zorder=2))
        # після штампа — червона рамка на решту життя
        if tid in stamp_at:
            sa = stamp_at[tid]
            tail = [f for f in fs if f >= sa]
            for a, b in _runs(np.array(tail)):
                ax1.add_patch(Rectangle((a, y - 0.32), b - a + 1, 0.64,
                              facecolor=RED, alpha=0.12, edgecolor=RED,
                              linewidth=0.8, zorder=3))
            ax1.scatter([sa], [y], marker='x', s=46, c=RED, zorder=6, linewidths=1.6)
            ax1.annotate('штамп', (sa, y), textcoords='offset points',
                         xytext=(4, 6), fontsize=7.5, color=RED, zorder=6)

    # траєкторії: де ціль (зелена) і що тримає трекер (синя)
    gt_xy = [(f, lane_y[s['gt_tid']]) for f, s in enumerate(snaps)
             if s and s['gt_tid'] in lane_y]
    held_xy = [(f, lane_y[s['held_tid']]) for f, s in enumerate(snaps)
               if s and s['held_tid'] in lane_y]
    if gt_xy:
        gx, gy = zip(*gt_xy)
        ax1.scatter(gx, gy, s=9, c=AQUA, zorder=5, label='де ЦІЛЬ (GT)')
    if held_xy:
        hx, hy = zip(*held_xy)
        ax1.scatter(hx, hy, s=9, c=BLUE, marker='s', zorder=5,
                    label='що тримає трекер (D14)')

    # рядок згорнутих ефемерних уламків: щільність присутності + їхні штампи
    if ephemeral:
        dens = np.zeros(n)
        eph_stamp = []
        for tid in ephemeral:
            for f in seen[tid]:
                dens[f] += 1
            if tid in stamp_at:
                eph_stamp.append(stamp_at[tid])
        if dens.max() > 0:
            for f in np.where(dens > 0)[0]:
                ax1.add_patch(Rectangle((f, eph_y - 0.34), 1, 0.68,
                              facecolor=MUTED, alpha=min(0.85, 0.15 + 0.7 * dens[f] / dens.max()),
                              edgecolor='none', zorder=1))
        if eph_stamp:
            ax1.scatter(eph_stamp, [eph_y] * len(eph_stamp), marker='x', s=20,
                        c=RED, alpha=0.5, zorder=5, linewidths=1.0)

    ax1.set_ylim(-0.8, H - 0.2)
    ax1.set_yticks(range(H))
    labels = [f't{t}' for t in order] + ([f'{len(ephemeral)} уламків'] if ephemeral else [])
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_ylabel('доріжки об’єктів (tracklet)')
    ax1.set_title(title, loc='left', color=INK, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(loc='lower right', frameon=False, fontsize=8, ncol=2)

    # ── Панель 2: IoU до GT, D14 vs D0 ─────────────────────────────────────
    _shade_absent(ax2, absent, frames)
    ax2.plot(frames, iou0, color=MUTED, lw=1.1, label='D0 (без вето)')
    ax2.plot(frames, iou14, color=PLUM, lw=1.4, label='D14 (вето)')
    ax2.axhline(0.5, color=AXIS, lw=0.7, ls='--')
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel('IoU до GT')
    ax2.legend(loc='upper right', frameon=False, fontsize=8, ncol=2)

    # ── Панель 3: події вето ───────────────────────────────────────────────
    _shade_absent(ax3, absent, frames)
    for f, s in enumerate(snaps):
        if s is None:
            continue
        vetoed = {tid for tid, d in s['live'].items() if d['matched'] and d['stamped']}
        if not vetoed:
            continue
        off = not (iou14[f] > 0.5)     # трекер не на цілі → момент re-detection
        if s['gt_tid'] in vetoed:      # заблокували справжню ціль
            ax3.axvline(f, color=RED, lw=0.6, alpha=0.8)
        elif off:                      # заблокували дистрактора під час пошуку
            ax3.axvline(f, color=AQUA, lw=0.5, alpha=0.5)
    ax3.set_yticks([])
    ax3.set_ylabel('вето', rotation=0, ha='right', va='center')
    ax3.set_xlabel('кадр')
    ax3.set_xlim(0, n - 1)
    handles = [Line2D([0], [0], color=AQUA, lw=2, label='заблокований дистрактор (користь)'),
               Line2D([0], [0], color=RED, lw=2, label='заблокована ЦІЛЬ (податок)')]
    ax3.legend(handles=handles, loc='upper right', frameon=False, fontsize=8, ncol=2)

    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f'🖼  {out}')


def _runs(fs):
    """Масив кадрів → суміжні відрізки (a, b) включно."""
    if len(fs) == 0:
        return
    fs = np.sort(fs)
    a = fs[0]
    for i in range(1, len(fs)):
        if fs[i] != fs[i - 1] + 1:
            yield a, fs[i - 1]
            a = fs[i]
    yield a, fs[-1]


def _shade_absent(ax, absent, frames):
    for a, b in _runs(frames[absent[:len(frames)]]):
        ax.axvspan(a, b + 1, color=GRID, alpha=0.6, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--data-dir', default='/home/peoly/datasets/lasot/test')
    ap.add_argument('--config14', default='yoloe-vp-iou/ablation/D14_coexist.yaml')
    ap.add_argument('--config0', default='yoloe-vp-iou/ablation/D0_minimal.yaml')
    ap.add_argument('--model', default='yoloe-v8m-seg.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--num-frames', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--title', default=None)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    vp = find_video(Path(args.data_dir), args.video)
    images = sorted((vp / 'img').glob('*.jpg'))
    gts = load_gt(vp / 'groundtruth.txt')[:len(images)]
    n = min(len(images), len(gts))
    if args.num_frames:
        n = min(n, args.num_frames)
    images, gts = images[:n], gts[:n]
    absent = load_flags(vp, 'full_occlusion.txt', n) | load_flags(vp, 'out_of_view.txt', n)

    import pickle
    cache = Path(f'/tmp/claude-1000/-home-peoly-Projects-basic-yoloe-tracker/'
                 f'f71e4986-2fdd-4834-977c-fdf16a17ca4f/scratchpad/coexist_viz_{args.video}.pkl')
    if cache.exists() and not args.no_cache:
        print(f'♻️  кеш: {cache}')
        iou14, iou0, snaps = pickle.load(open(cache, 'rb'))
    else:
        print(f'📹 {args.video}: {n} кадрів — D14 (з пулом) + D0 (для порівняння)')
        iou14, snaps = run_tracker(vp, images, gts, args.config14, args.model,
                                   args.imgsz, capture_pool=True)
        iou0, _ = run_tracker(vp, images, gts, args.config0, args.model,
                              args.imgsz, capture_pool=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((iou14, iou0, snaps), open(cache, 'wb'))

    out = args.out or f'figures/coexist/{args.video}.png'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    a14 = np.nanmean([x for x in iou14 if not np.isnan(x)])
    a0 = np.nanmean([x for x in iou0 if not np.isnan(x)])
    title = args.title or f'{args.video} — MOT-виключення | D0 {a0:.3f} → D14 {a14:.3f}'
    build(args.video, snaps, np.array(iou14), np.array(iou0), absent, title, out)


if __name__ == '__main__':
    main()
