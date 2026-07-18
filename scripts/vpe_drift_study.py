#!/usr/bin/env python3
"""
Дрейф пам'яті VPE відносно ground-truth вигляду цілі, по кадрах.

Питання: чи взагалі пам'ять тримає ціль? На кожному кадрі кодуємо VPE з GT-боксу
(незалежний референс — те, як ціль виглядає НАСПРАВДІ зараз) і міряємо косинус до
кожного представлення пам'яті трекера:

  anchor  — VPE кадру 0, фіксований
  LT      — середнє long-term банку
  ST      — temporal-decay середнє short-term вікна
  agg     — фінальна агрегація (те, що реально йде у set_classes)

Ключова відмінність від наявного `anchor_proximity` у трекері: там пам'ять
міряється проти ПОТОЧНОЇ ДЕТЕКЦІЇ, тож після зльоту на дистрактор схожість
лишається високою. Тут референс — GT, тож дрейф видно як реальне падіння.

Нижня панель — per-frame IoU трекера, щоб зіставити падіння косинуса з втратою.

Використання:
  python scripts/vpe_drift_study.py --video book-19 \
         --tracker-config ablation/D0_minimal.yaml \
         [--data-dir /home/peoly/datasets/lasot/test] [--stride 2] \
         [--out figures/vpe_drift_book-19.png] [--csv out.csv]

Прогонити конда-оточенням tracking_dev:
  /home/peoly/anaconda3/envs/tracking_dev/bin/python scripts/vpe_drift_study.py ...
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from trackers import TrackerRegistry            # noqa: E402
from config_loader import ConfigLoader          # noqa: E402

BLUE, RED, AQUA, PLUM = '#2a78d6', '#e34948', '#0f9068', '#8a4fbd'
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

TRACKS = [('anchor', BLUE), ('LT', AQUA), ('ST', RED), ('agg', PLUM)]

# Фази трекера (та сама логіка, що в get_debug_info: чиста функція від lost_frames)
AMBER, ROSE = '#d98a1f', '#c94b52'
PHASES = {1: ('Phase 1: IoU', '#cfd8cf'), 2: ('Phase 2: очікування', AMBER),
          3: ('Phase 3: ре-ID', ROSE)}

# Стан за ground-truth — те, що є НАСПРАВДІ, незалежно від думки трекера.
ON, DRIFT, LOST, UNK = 0, 1, 2, 3
GT_STATES = {ON: ('на цілі (IoU>0.5)', '#0f9068'), DRIFT: ('дрейф (0.1–0.5)', '#d9b84f'),
             LOST: ('не на цілі (IoU<0.1)', '#c0392b'), UNK: ('немає даних', '#d5d4cd')}


def gt_state(i: float) -> int:
    """Реальний стан за IoU до GT. nan (нема GT або трекер не дав бокс) → UNK."""
    if np.isnan(i):
        return UNK
    return ON if i > 0.5 else (LOST if i < 0.1 else DRIFT)


def phase_of(tracker) -> int:
    """1 — трек живий, 2 — втрачено але ще чекаємо, 3 — ре-ідентифікація."""
    lost = getattr(tracker, 'lost_frames', 0)
    if lost == 0:
        return 1
    return 2 if lost < getattr(tracker, 'max_lost_frames', 1) else 3


def runs(values):
    """Послідовність → (i_start, i_end_inclusive, value) для суміжних однакових ділянок."""
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            yield start, i - 1, values[start]
            start = i


def find_video(data_dir: Path, video: str) -> Path:
    """LaSOT: <data>/<class-dir>/<class>-<n>/. Приймає і повний шлях.

    Каталог класу на диску може мати префікс (lasot_test_book), тож шукаємо глобом.
    """
    p = Path(video)
    if p.is_dir():
        return p
    hits = [d for d in data_dir.glob(f'*/{video}') if (d / 'groundtruth.txt').exists()]
    if not hits:
        raise SystemExit(f'❌ Не знайдено послідовність «{video}» у {data_dir}')
    return hits[0]


def load_gt(gt_file: Path):
    """groundtruth.txt → список [x, y, w, h] або None."""
    out = []
    for line in gt_file.read_text().splitlines():
        line = line.strip()
        if not line or 'nan' in line.lower():
            out.append(None)
            continue
        parts = line.split(',')
        try:
            out.append([float(p) for p in parts[:4]] if len(parts) >= 4 else None)
        except ValueError:
            out.append(None)
    return out


def load_flags(video_path: Path, name: str, n: int):
    """full_occlusion.txt / out_of_view.txt → bool-масив довжини n (нема файлу → усе False)."""
    f = video_path / name
    if not f.exists():
        return np.zeros(n, dtype=bool)
    vals = f.read_text().replace('\n', ',').split(',')
    flags = [v.strip() == '1' for v in vals if v.strip() != '']
    flags = (flags + [False] * n)[:n]
    return np.array(flags, dtype=bool)


def iou(a, b) -> float:
    """Обидва бокси у форматі [x, y, w, h] — так само, як їх трактує eval
    (tracker.update повертає xywh, не xyxy)."""
    if a is None or b is None:
        return np.nan
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else np.nan


def flat_unit(v):
    """VPE будь-якого shape → L2-нормований 1-D float-тензор на CPU."""
    if v is None or not isinstance(v, torch.Tensor):
        return None
    return F.normalize(v.detach().flatten().float().cpu(), p=2, dim=0)


def cos(a, b) -> float:
    """Сирий косинус (НЕ (cos+1)/2, як у _calculate_cosine_similarity трекера)."""
    if a is None or b is None:
        return np.nan
    return float(torch.dot(a, b))


def box_vpe(tracker, image, xywh):
    """Закодувати VPE з довільного боксу [x,y,w,h], не торкаючись пам'яті трекера."""
    if xywh is None:
        return None
    x, y, w, h = xywh
    if w <= 1 or h <= 1:
        return None
    try:
        vp = tracker._get_vpe_predictor()
        vp.set_prompts(dict(bboxes=np.array([[x, y, x + w, y + h]], dtype=np.float32),
                            cls=np.array([0])))
        return flat_unit(vp.get_vpe(image))
    except Exception as e:
        print(f'   ⚠️  box_vpe: {e}')
        return None


def memory_views(tracker):
    """Живий стан пам'яті → {назва: одиничний вектор}."""
    dm = getattr(tracker, 'dual_memory', None)
    if dm is None:
        return {k: None for k, _ in TRACKS}
    return {
        'anchor': flat_unit(dm.get_anchor_vpe()),
        'LT': flat_unit(dm.get_long_term_avg_vpe()),
        'ST': flat_unit(dm.get_short_term_avg_vpe()),
        'agg': flat_unit(dm.get_aggregated_vpe()),
    }


def run(args):
    video_path = find_video(Path(args.data_dir), args.video)
    name = video_path.name
    images = sorted((video_path / 'img').glob('*.jpg'))
    gts = load_gt(video_path / 'groundtruth.txt')[:len(images)]
    if args.num_frames:
        images, gts = images[:args.num_frames], gts[:args.num_frames]
    n = min(len(images), len(gts))
    images, gts = images[:n], gts[:n]
    if not n or gts[0] is None:
        raise SystemExit('❌ Порожня послідовність або невалідний GT кадру 0')

    params = {'model_path': args.model, 'imgsz': args.imgsz}
    if args.tracker_config:
        params.update(ConfigLoader().load(args.tracker_config))
    tracker = TrackerRegistry.get_tracker(args.tracker, **params)

    print(f'📹 {name}: {n} кадрів | конфіг: {args.tracker_config or "defaults"} '
          f'| GT-probe кожні {args.stride} кадр(и)')

    rows = []
    for idx, img_path in enumerate(tqdm(images, desc='drift', leave=False)):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        gt = gts[idx]

        if idx == 0:
            ok = tracker.initialize(image, gt)
            pred = gt if ok else None
        else:
            ok, pred = tracker.update(image)
            if not ok:
                pred = None

        if idx % args.stride:
            continue

        g = box_vpe(tracker, image, gt) if (gt is not None and not args.no_gt_probe) else None
        mem = memory_views(tracker)
        row = dict(frame=idx, iou=iou(pred, gt), phase=phase_of(tracker),
                   **{k: cos(mem[k], g) for k, _ in TRACKS})
        # бокс трекера — щоб рахувати IoU між сусідніми кадрами (те, з чим
        # порівнюється iou_threshold на Phase-1 матчі)
        row.update(dict(zip(('px', 'py', 'pw', 'ph'),
                            pred if pred is not None else (np.nan,) * 4)))
        # cos(пам'ять, VPE ТОГО боксу, який тримає трекер) — саме цю величину
        # міг би бачити identity-гейт у реальному часі (GT він не має).
        if not args.no_gt_probe:
            p = box_vpe(tracker, image, pred)
            row.update({f'pred_{k}': cos(mem[k], p) for k, _ in TRACKS})
        rows.append(row)

    if not rows:
        raise SystemExit('❌ Жодного кадру не оброблено')

    frames = np.array([r['frame'] for r in rows])
    ious = np.array([r['iou'] for r in rows], dtype=float)
    phases = np.array([r['phase'] for r in rows], dtype=int)
    curves = {k: np.array([r[k] for r in rows], dtype=float) for k, _ in TRACKS}
    absent = load_flags(video_path, 'full_occlusion.txt', n) | \
        load_flags(video_path, 'out_of_view.txt', n)

    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = (['frame', 'iou', 'phase'] + [k for k, _ in TRACKS]
                + [c for c in (f'pred_{k}' for k, _ in TRACKS) if c in rows[0]]
                + ['px', 'py', 'pw', 'ph'])
        lines = [','.join(cols)]
        lines += [','.join(f'{r[c]:.6f}' if isinstance(r[c], float) else str(r[c])
                           for c in cols) for r in rows]
        out_csv.write_text('\n'.join(lines) + '\n')
        print(f'💾 CSV: {out_csv}')

    print(f'\n{"вид":<8}{"cos сер":>10}{"cos мін":>10}{"кінець":>10}{"Δ від старту":>15}')
    for k, _ in TRACKS:
        c = curves[k]
        if np.all(np.isnan(c)):
            continue
        first = c[~np.isnan(c)][0]
        last = c[~np.isnan(c)][-1]
        print(f'{k:<8}{np.nanmean(c):>10.3f}{np.nanmin(c):>10.3f}'
              f'{last:>10.3f}{last - first:>+15.3f}')
    print(f'\nсередній IoU: {np.nanmean(ious):.3f}')
    for p in (2, 3):
        share = float((phases == p).mean())
        entries = sum(1 for _, _, v in runs(list(phases)) if v == p)
        print(f'{PHASES[p][0]:<22}{share:6.1%} кадрів, входів: {entries}')

    # Перетин «що трекер думає» × «що є насправді». Phase 1 при IoU<0.1 —
    # сліпий лок: трекер упевнено веде ЧУЖИЙ об'єкт і не шукає ціль.
    states = np.array([gt_state(i) for i in ious])
    blind = (phases == 1) & (states == LOST)
    n_lost = int((states == LOST).sum())
    if n_lost:
        print(f'сліпий лок: {blind.sum()}/{n_lost} кадрів поза ціллю трекер веде у Phase 1 '
              f'({blind.sum() / n_lost:.0%})')

    plot(args, name, frames, curves, ious, absent, phases, states, blind)


def plot(args, name, frames, curves, ious, absent, phases, states, blind):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(11, 7.0), sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1, 0.16, 0.16], hspace=0.12))

    for k, color in TRACKS:
        c = curves[k]
        if np.all(np.isnan(c)):
            continue
        ax1.plot(frames, c, color=color, lw=1.4 if k != 'agg' else 2.0,
                 alpha=0.9, label=k, zorder=3 if k == 'agg' else 2)

    ax1.set_ylabel('cos(пам\'ять, VPE з GT-боксу)')
    ax1.set_title(f'{name} — дрейф пам\'яті VPE від справжнього вигляду цілі', loc='left')
    ax1.grid(True, color=GRID, lw=0.6, zorder=0)
    ax1.legend(frameon=False, ncol=4, loc='lower left', fontsize=8.5)

    ax2.plot(frames, ious, color=INK2, lw=1.2, zorder=3)
    ax2.fill_between(frames, 0, ious, color=INK2, alpha=0.12, zorder=2)
    ax2.axhline(0.5, color=RED, lw=0.8, ls='--', alpha=0.6, zorder=1)
    ax2.set_ylabel('IoU трекера')
    ax2.set_ylim(0, 1)
    ax2.grid(True, color=GRID, lw=0.6, zorder=0)

    # затінити ділянки, де цілі нема (повна оклюзія / поза кадром)
    if absent.any():
        edges = np.diff(np.concatenate(([0], absent.astype(int), [0])))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            for ax in (ax1, ax2):
                ax.axvspan(s, e, color=MUTED, alpha=0.10, lw=0, zorder=1)

    def bounds(i, j):
        """Межі суміжної ділянки у координатах кадрів (роздільність = --stride)."""
        lo = frames[i] if i == 0 else (frames[i - 1] + frames[i]) / 2
        hi = frames[j] if j == len(frames) - 1 else (frames[j] + frames[j + 1]) / 2
        return lo, hi

    def ribbon(ax, values, palette, label):
        for i, j, v in runs(list(values)):
            lo, hi = bounds(i, j)
            ax.axvspan(lo, hi, color=palette[v][1], lw=0,
                       alpha=0.55 if v in (1, ON) else 0.95)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=8)
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)

    # Дві стрічки одна під одною: що трекер ДУМАЄ vs що є НАСПРАВДІ.
    # Розбіжність читається вертикально: сіре зверху + червоне знизу = сліпий лок.
    ribbon(ax3, phases, PHASES, 'фаза')
    ribbon(ax4, states, GT_STATES, 'GT')
    ax4.set_xlabel('кадр')

    # сліпий лок — підсвітити на панелі IoU, це головна патологія
    for i, j, v in runs(list(blind)):
        if v:
            lo, hi = bounds(i, j)
            ax2.axvspan(lo, hi, color=ROSE, alpha=0.16, lw=0, zorder=1)

    # вертикальні риски входу у фази 2/3
    for i, j, p in runs(list(phases)):
        if p in (2, 3) and i > 0:
            lo, _ = bounds(i, j)
            for ax in (ax1, ax2):
                ax.axvline(lo, color=PHASES[p][1], lw=0.7, alpha=0.5, zorder=1)

    keys = ([(PHASES, p) for p in (1, 2, 3) if (phases == p).any()]
            + [(GT_STATES, s) for s in (ON, DRIFT, LOST, UNK) if (states == s).any()])
    ax4.legend([plt.Rectangle((0, 0), 1, 1, color=pal[v][1],
                              alpha=0.55 if v in (1, ON) else 0.95) for pal, v in keys],
               [pal[v][0] for pal, v in keys], frameon=False, fontsize=8,
               ncol=4, loc='upper left', bbox_to_anchor=(0, -1.2))

    out = Path(args.out or f'figures/vpe_drift_{name}.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'🖼  Фігура: {out}  (сірі смуги = ціль відсутня)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True, help='напр. book-19, або повний шлях')
    ap.add_argument('--data-dir', default='/home/peoly/datasets/lasot/test')
    ap.add_argument('--tracker', default='YOLOe-VP-IoU')
    ap.add_argument('--tracker-config', default='yoloe-vp-iou/ablation/D0_minimal.yaml',
                    help='шлях БЕЗ префікса configs/ (loader додає сам)')
    ap.add_argument('--model', default='yoloe-v8m-seg.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--stride', type=int, default=2,
                    help='рахувати GT-VPE кожні N кадрів (трекер іде щокадру)')
    ap.add_argument('--num-frames', type=int, default=0)
    ap.add_argument('--no-gt-probe', action='store_true',
                    help='діагностика: не кодувати GT-VPE (перевірка, чи проба не псує трекер)')
    ap.add_argument('--out', help='шлях до PNG')
    ap.add_argument('--csv', help='зберегти сирі значення')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
