#!/usr/bin/env python3
"""
Оракульна діагностика: чи врятувала б компенсація руху камери (CMC) хибні втрати?

Питання. Усі гейти трекера — це IoU/DIoU між `last_valid_bbox` з кадру t-1 і
детекціями кадру t. Коли камера різко рухається, нерухома ціль зміщується в
координатах зображення, IoU падає нижче `iou_threshold`, і трекер оголошує
втрату, хоча ціль нікуди не зникала. CMC (як у BoT-SORT/StrongSORT) оцінює
глобальне перетворення фону і переносить опорну рамку в координати нового кадру.

Скрипт НЕ чіпає трекер. Він рахує верхню межу користі:

  подія втрати  — ціль присутня на t-1 і t, але IoU(gt[t-1], gt[t]) < поріг,
                  тобто навіть ІДЕАЛЬНА детекція цілі не пройшла б гейт Фази 1;
  врятовано     — після переносу gt[t-1] глобальним перетворенням IoU >= поріг;
  зіпсовано     — здоровий кадр (IoU >= поріг), який після переносу впав нижче.

Частка «врятовано» серед подій — стеля того, що CMC може дати. Низька частка
означає, що втрати спричинені рухом самої цілі чи перекриттям, і кодувати
компенсацію немає сенсу (як це вже сталось із Kalman і LK-референсом).

Перетворення оцінюється ORB+RANSAC (partial affine) по ФОНУ: область цілі
маскується, щоб сама ціль не тягнула оцінку на себе.

Використання:
  python scripts/camera_motion_study.py --test-list lasot_probe16.txt
  python scripts/camera_motion_study.py --video cattle-2 --csv out.csv

Залежності лише cv2/numpy — трекер і torch не потрібні, годиться будь-яке
оточення з opencv-contrib.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_DATA = '/home/peoly/datasets/lasot/test'


# ─────────────────────────── завантаження послідовності ──────────────────────

def resolve_video(data_dir: Path, video: str) -> Path:
    """Каталог класу на диску має префікс (lasot_test_cattle), тож шукаємо глобом."""
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
    """full_occlusion.txt / out_of_view.txt → bool-масив довжини n."""
    f = video_path / name
    if not f.exists():
        return np.zeros(n, dtype=bool)
    vals = f.read_text().replace('\n', ',').split(',')
    flags = [v.strip() == '1' for v in vals if v.strip() != '']
    flags = (flags + [False] * n)[:n]
    return np.array(flags, dtype=bool)


# ────────────────────────────── геометрія боксів ─────────────────────────────

def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def diou(a, b) -> float:
    """DIoU у тій самій формі, що й у трекері: IoU мінус нормована відстань центрів."""
    base = iou(a, b)
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    rho2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    cx1, cy1 = min(a[0], b[0]), min(a[1], b[1])
    cx2, cy2 = max(a[2], b[2]), max(a[3], b[3])
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    return base - rho2 / c2 if c2 > 0 else base


def compose(A, B):
    """Композиція афінних 2x3: результат застосовує спершу B, потім A."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def warp_box(box_xyxy, M) -> np.ndarray:
    """Перенести бокс перетворенням 2x3; повертає осе-вирівняну оболонку."""
    x1, y1, x2, y2 = box_xyxy
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    warped = (M[:, :2] @ pts.T).T + M[:, 2]
    return np.array([warped[:, 0].min(), warped[:, 1].min(),
                     warped[:, 0].max(), warped[:, 1].max()])


# ──────────────────────────── оцінка руху камери ─────────────────────────────

class GlobalMotion:
    """ORB+RANSAC оцінка глобального (фонового) перетворення між кадрами."""

    def __init__(self, n_features: int = 1000, min_matches: int = 12,
                 mask_dilate: float = 0.25):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.mask_dilate = mask_dilate

    def _mask(self, shape, box_xyxy):
        """255 усюди, 0 — у роздутій області цілі (щоб ціль не тягнула оцінку)."""
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        if box_xyxy is None:
            return mask
        x1, y1, x2, y2 = box_xyxy
        dw, dh = (x2 - x1) * self.mask_dilate, (y2 - y1) * self.mask_dilate
        x1, y1 = int(max(0, x1 - dw)), int(max(0, y1 - dh))
        x2, y2 = int(min(shape[1], x2 + dw)), int(min(shape[0], y2 + dh))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
        return mask

    def estimate(self, prev_gray, prev_box, cur_gray, cur_box):
        """Повертає (M 2x3, n_inliers) або (None, 0), якщо оцінити не вдалось."""
        kp1, des1 = self.orb.detectAndCompute(prev_gray, self._mask(prev_gray.shape, prev_box))
        kp2, des2 = self.orb.detectAndCompute(cur_gray, self._mask(cur_gray.shape, cur_box))
        if des1 is None or des2 is None or len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return None, 0
        matches = self.matcher.match(des1, des2)
        if len(matches) < self.min_matches:
            return None, 0
        src = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst = np.float32([kp2[m.trainIdx].pt for m in matches])
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=2000, confidence=0.99)
        if M is None:
            return None, 0
        return M, int(inliers.sum()) if inliers is not None else 0


# ──────────────────────────────── аналіз відео ───────────────────────────────

def analyse(video_path: Path, args):
    images = sorted((video_path / 'img').glob('*.jpg'))
    gts = load_gt(video_path / 'groundtruth.txt')[:len(images)]
    n = min(len(images), len(gts))
    absent = load_flags(video_path, 'full_occlusion.txt', n) | \
        load_flags(video_path, 'out_of_view.txt', n)

    gm = GlobalMotion(min_matches=args.min_matches)
    rows, lag_rows = [], []
    prev_gray = prev_box = None
    # історія для режиму «стухлої опорної рамки» (Фаза 3): бокси і накопичені
    # перетворення від кадру t-k до поточного
    hist_boxes, hist_M = [], []

    it = range(0, n, args.stride)
    for i in tqdm(list(it), desc=video_path.name, leave=False, disable=args.quiet):
        gt = gts[i]
        box = xywh_to_xyxy(gt) if (gt is not None and not absent[i]) else None
        gray = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            prev_gray, prev_box = None, None
            hist_boxes, hist_M = [], []
            continue

        if prev_gray is not None and prev_box is not None and box is not None:
            M, inl = gm.estimate(prev_gray, prev_box, gray, box)
            raw_iou, raw_diou = iou(prev_box, box), diou(prev_box, box)
            if M is None:
                rows.append((i, raw_iou, raw_diou, np.nan, np.nan, np.nan, inl))
                hist_boxes, hist_M = [], []       # розрив ланцюга накопичення
            else:
                wb = warp_box(prev_box, M)
                shift = float(np.hypot(M[0, 2], M[1, 2]))
                rows.append((i, raw_iou, raw_diou, iou(wb, box), diou(wb, box), shift, inl))

                # ── Фаза 3: опорна рамка застаріла на args.lag кадрів ──
                hist_M = [compose(M, acc) for acc in hist_M] + [M]
                hist_boxes.append(prev_box)
                if len(hist_boxes) > args.lag:
                    hist_boxes.pop(0)
                    hist_M.pop(0)
                if len(hist_boxes) == args.lag:
                    stale, acc = hist_boxes[0], hist_M[0]
                    lag_rows.append((i, diou(stale, box),
                                     diou(warp_box(stale, acc), box)))
        else:
            hist_boxes, hist_M = [], []

        prev_gray, prev_box = gray, box

    return rows, lag_rows


def summarise(name, rows, lag_rows, args):
    """Події втрати / порятунки / псування + скільки з цього пояснює рух камери."""
    ok = [r for r in rows if not np.isnan(r[3])]
    if not ok:
        return None
    # Фаза 3: скільки разів СПРАВЖНЯ ціль відкидається anti-teleport floor
    # через застарілу опорну рамку, і скільки з них рятує компенсація
    floor = args.diou_floor
    p3_events = [r for r in lag_rows if r[1] < floor]
    p3_rescued = [r for r in p3_events if r[2] >= floor]
    p3_harmed = [r for r in lag_rows if r[1] >= floor and r[2] < floor]
    thr = args.iou_threshold
    events = [r for r in ok if r[1] < thr]
    rescued = [r for r in events if r[3] >= thr]
    healthy = [r for r in ok if r[1] >= thr]
    harmed = [r for r in healthy if r[3] < thr]
    # Фаза 2 працює по DIoU з іншим порогом
    p2_events = [r for r in ok if r[2] < args.diou_threshold]
    p2_rescued = [r for r in p2_events if r[4] >= args.diou_threshold]
    shifts = np.array([r[5] for r in ok])
    return {
        'video': name,
        'pairs': len(ok),
        'unestimable': len(rows) - len(ok),
        'events': len(events),
        'rescued': len(rescued),
        'rescue_frac': len(rescued) / len(events) if events else float('nan'),
        'harmed': len(harmed),
        'harm_frac': len(harmed) / len(healthy) if healthy else float('nan'),
        'p2_events': len(p2_events),
        'p2_rescued': len(p2_rescued),
        'p3_pairs': len(lag_rows),
        'p3_events': len(p3_events),
        'p3_rescued': len(p3_rescued),
        'p3_harmed': len(p3_harmed),
        'shift_med': float(np.median(shifts)),
        'shift_p90': float(np.percentile(shifts, 90)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--data-dir', default=DEFAULT_DATA)
    ap.add_argument('--video', help='одна послідовність (напр. cattle-2)')
    ap.add_argument('--test-list', help='файл зі списком послідовностей')
    ap.add_argument('--stride', type=int, default=1,
                    help='крок по кадрах (1 = кожен; більший = швидше, грубіше)')
    ap.add_argument('--iou-threshold', type=float, default=0.2,
                    help='гейт Фази 1 трекера (iou_threshold)')
    ap.add_argument('--diou-threshold', type=float, default=-0.1,
                    help='гейт Фази 2 трекера (phase2_diou_threshold)')
    ap.add_argument('--lag', type=int, default=15,
                    help='на скільки кадрів застаріває опорна рамка у Фазі 3 '
                         '(= max_lost_frames трекера)')
    ap.add_argument('--diou-floor', type=float, default=-0.7,
                    help='anti-teleport floor Фази 3 (reinit_diou_floor)')
    ap.add_argument('--min-matches', type=int, default=12)
    ap.add_argument('--csv', help='записати покадрові дані сюди')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.video:
        names = [args.video]
    elif args.test_list:
        names = [l.split('#')[0].strip() for l in Path(args.test_list).read_text().splitlines()
                 if l.strip() and not l.strip().startswith('#')]
        names = [n for n in names if n]
    else:
        raise SystemExit('❌ Потрібен --video або --test-list')

    all_rows, summaries = [], []
    for name in names:
        vp = resolve_video(data_dir, name)
        rows, lag_rows = analyse(vp, args)
        s = summarise(name, rows, lag_rows, args)
        if s is None:
            print(f'⚠️  {name}: не вдалось оцінити жодної пари')
            continue
        summaries.append(s)
        all_rows += [(name,) + r for r in rows]
        p3f = s['p3_rescued'] / s['p3_events'] if s['p3_events'] else float('nan')
        print(f"{s['video']:16} Ф1: події={s['events']:5d} рят={s['rescued']:5d} "
              f"({s['rescue_frac']:6.1%}) | Ф3(lag{args.lag}): події={s['p3_events']:5d} "
              f"рят={s['p3_rescued']:5d} ({p3f:6.1%}) шкода={s['p3_harmed']:4d} | "
              f"зсув med={s['shift_med']:5.1f}px p90={s['shift_p90']:6.1f}px")

    if not summaries:
        return

    ev = sum(s['events'] for s in summaries)
    rs = sum(s['rescued'] for s in summaries)
    hm = sum(s['harmed'] for s in summaries)
    p2e = sum(s['p2_events'] for s in summaries)
    p2r = sum(s['p2_rescued'] for s in summaries)
    pairs = sum(s['pairs'] for s in summaries)
    print('\n' + '=' * 78)
    print(f"ПІДСУМОК ({len(summaries)} відео, {pairs} пар кадрів)")
    print(f"  Фаза 1 (IoU < {args.iou_threshold}):   подій {ev:6d} | "
          f"CMC рятує {rs:6d} = {rs / ev:.1%}" if ev else "  Фаза 1: подій немає")
    print(f"  Фаза 2 (DIoU < {args.diou_threshold}): подій {p2e:6d} | "
          f"CMC рятує {p2r:6d} = {p2r / p2e:.1%}" if p2e else "  Фаза 2: подій немає")
    p3e = sum(s['p3_events'] for s in summaries)
    p3r = sum(s['p3_rescued'] for s in summaries)
    p3h = sum(s['p3_harmed'] for s in summaries)
    if p3e:
        print(f"  Фаза 3 (стухла рамка, lag={args.lag}, floor={args.diou_floor}): "
              f"подій {p3e:6d} | CMC рятує {p3r:6d} = {p3r / p3e:.1%} | шкода {p3h}")
    print(f"  Побічна шкода Фази 1: {hm} здорових кадрів впали б нижче порога")
    print('=' * 78)
    print('Читання: висока частка порятунку → CMC має стелю і варта реалізації;')
    print('низька → втрати від руху цілі/перекриття, компенсація буде інертна.')

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video', 'frame', 'iou_raw', 'diou_raw',
                        'iou_cmc', 'diou_cmc', 'shift_px', 'inliers'])
            w.writerows(all_rows)
        print(f'\n📄 Покадрові дані: {args.csv}')


if __name__ == '__main__':
    main()
