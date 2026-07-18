#!/usr/bin/env python3
"""
Податок правила «виключення за співіснуванням» (MOT-шар), по кадрах.

Правило-кандидат: об'єкт, ВИДИМИЙ ОДНОЧАСНО з ціллю у роз'єднаній позиції, доведено
НЕ є ціллю → його бокс назавжди неприйнятний для re-detection. Це єдиний механізм
проєкту, що не порівнює зовнішність.

Зворотне tracklet-дослідження (2026-07-16) міряло ЛИШЕ користь на 4 hijack-відео.
Тут міряється ПОДАТОК — як часто правило викреслить СПРАВЖНЮ ціль:

  ПОДАТОК  — кадри, де ціль є серед детекцій (IoU з GT > 0.5), але її tracklet
             уже проштампований «не ціль» → правило заблокує правильний re-lock.
  КОРИСТЬ  — кадри, де трекер тримає чужий бокс, а той проштампований → правило
             завадило б цьому локу.

Два джерела штампа рахуються паралельно:
  proxy  — бокс трекера (те, що доступно у рантаймі; отруюється сліпим локом)
  oracle — GT-бокс (верхня межа: правило з ідеальним знанням, де ціль)
Якщо податок великий навіть в oracle — вісь мертва в принципі. Якщо oracle чистий,
а proxy отруєний — правило вимагає спершу розв'язати сліпий лок (кругова залежність).

Штампи класифікуються за станом трекера У МОМЕНТ штампування:
  clean — трекер був НА ЦІЛІ (IoU>0.5) → штамп на цілі = правило зламане в принципі
  dirty — трекер уже був не на цілі → отруєння, наслідок наявного провалу

Використання:
  python scripts/coexist_exclusion_study.py --video book-19 [--csv out.csv]
  python scripts/coexist_exclusion_study.py --videos lasot_probe16.txt --csv-dir results_coexist/

Прогонити конда-оточенням tracking_dev:
  /home/peoly/anaconda3/envs/tracking_dev/bin/python scripts/coexist_exclusion_study.py ...
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from trackers import TrackerRegistry            # noqa: E402
from config_loader import ConfigLoader          # noqa: E402
from vpe_drift_study import find_video, load_gt, iou  # noqa: E402

CLEAN, DIRTY = 'clean', 'dirty'


def xywh_to_xyxy(b):
    return None if b is None else [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def iou_xyxy(a, b) -> float:
    """IoU двох боксів xyxy. None → nan."""
    if a is None or b is None:
        return np.nan
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else np.nan


class Tracklet:
    __slots__ = ('id', 'box', 'misses', 'born', 'stamp', 'disjoint', 'vel')

    def __init__(self, tid, box, frame):
        self.id = tid
        self.box = box
        self.misses = 0
        self.born = frame
        # {'proxy': (frame, kind) | None, 'oracle': (frame, kind) | None}
        self.stamp = {'proxy': None, 'oracle': None}
        # накопичені кадри роз'єднаного співіснування з ціллю (доказ «не ціль»)
        self.disjoint = {'proxy': 0, 'oracle': 0}
        # покадровий зсув боксу (SORT-lite предикція за постійною швидкістю)
        self.vel = np.zeros(4, dtype=np.float32)

    def predicted(self):
        """Бокс, екстрапольований на misses+1 кадрів уперед за останньою швидкістю.
        misses=0 → сам box; коастинг зсуває його туди, куди об'єкт МАВ би дійти."""
        return (np.asarray(self.box, dtype=np.float32)
                + self.vel * (self.misses + 1)).tolist()


class TrackletPool:
    """SORT-lite: асоціація ТІЛЬКИ за IoU, без Kalman і без зовнішності.

    Kalman фальсифіковано у 4 ролях, зовнішність не розрізняє екземпляри — тож
    навмисно найдешевший шар, який лише встановлює факт «це той самий об'єкт».
    """

    def __init__(self, link_thr: float, max_age: int, predict: bool = False):
        self.link_thr = link_thr
        self.max_age = max_age
        self.predict = predict          # SORT-lite: матчити проти передбаченого боксу
        self.tracks = []
        self._next_id = 0

    def step(self, dets, frame: int):
        """dets — список xyxy. Повертає {tracklet_id: det_index} за цей кадр."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            ref = t.predicted() if self.predict else t.box
            for di, d in enumerate(dets):
                v = iou_xyxy(ref, d)
                if not np.isnan(v) and v >= self.link_thr:
                    pairs.append((v, ti, di))
        pairs.sort(key=lambda p: -p[0])

        used_t, used_d, owner = set(), set(), {}
        for v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            new_box = np.asarray(dets[di], dtype=np.float32)
            if self.predict:
                # швидкість = зсув за (misses+1) кадрів з останнього матчу, EMA-згладжена
                inst = (new_box - np.asarray(t.box, dtype=np.float32)) / (t.misses + 1)
                t.vel = 0.5 * t.vel + 0.5 * inst
            t.box = dets[di]
            t.misses = 0
            owner[t.id] = di

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        for di, d in enumerate(dets):
            if di not in used_d:
                t = Tracklet(self._next_id, d, frame)
                self._next_id += 1
                self.tracks.append(t)
                owner[t.id] = di
        return owner


def stamp_pass(pool, owner, dets, ref_box, frame, kind, variant, min_frames=1):
    """Проштампувати «не ціль» tracklet-и, що ДОВГО існують окремо від ref_box.

    Правило спирається на твердження «об'єкт, який співіснує з ціллю роз'єднано,
    не є ціллю». Одного кадру роз'єднаності для цього НЕ досить: одна помилка
    IoU-асоціації дає вічний хибний штамп (sepia-13: штамп на кадрі 1 отруїв
    2116 з 2713 кадрів). Тому потрібно min_frames накопичених кадрів доказу.

    Якщо tracklet колись зматчився з ціллю (IoU>0.5) — доказ спростовано:
    лічильник і штамп скидаються.

    ref_box відсутній (трекер без боксу / нема GT) → штампувати НЕМА ЧОГО: факт
    співіснування не встановлено. Саме тому провал у 16-кадровій дірі book-19 не
    псує штамп, поставлений раніше.
    """
    if ref_box is None:
        return
    # tracklet самої цілі (той, що володіє ref_box)
    own = None
    best = 0.5
    for t in pool.tracks:
        if t.id not in owner:
            continue
        v = iou_xyxy(dets[owner[t.id]], ref_box)
        if not np.isnan(v) and v > best:
            best, own = v, t.id
    for t in pool.tracks:
        if t.id not in owner:
            continue
        if t.id == own:                       # це ціль — доказ «не ціль» спростовано
            t.disjoint[variant] = 0
            t.stamp[variant] = None
            continue
        if t.stamp[variant] is not None:
            continue
        v = iou_xyxy(dets[owner[t.id]], ref_box)
        if not np.isnan(v) and v <= 0.0:      # роз'єднано — кадр доказу
            t.disjoint[variant] += 1
            if t.disjoint[variant] >= min_frames:
                t.stamp[variant] = (frame, kind)


def run_video(args, video: str):
    video_path = find_video(Path(args.data_dir), video)
    images = sorted((video_path / 'img').glob('*.jpg'))
    gts = load_gt(video_path / 'groundtruth.txt')[:len(images)]
    n = min(len(images), len(gts))
    if args.num_frames:
        n = min(n, args.num_frames)
    images, gts = images[:n], gts[:n]
    if not n or gts[0] is None:
        raise SystemExit(f'❌ {video}: порожня послідовність або невалідний GT кадру 0')

    params = {'model_path': args.model, 'imgsz': args.imgsz}
    if args.tracker_config:
        params.update(ConfigLoader().load(args.tracker_config))
    tracker = TrackerRegistry.get_tracker(args.tracker, **params)

    pool = TrackletPool(args.link_thr, args.max_age, predict=args.predict)
    rows = []

    for idx, img_path in enumerate(tqdm(images, desc=video, leave=False)):
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

        raw = getattr(tracker, 'last_detections', None)
        dets = [] if raw is None or len(raw) == 0 else [list(r[:4]) for r in raw]

        pred_xyxy, gt_xyxy = xywh_to_xyxy(pred), xywh_to_xyxy(gt)
        tr_iou = iou(pred, gt)                       # IoU трекера з GT (xywh)
        on_target = (not np.isnan(tr_iou)) and tr_iou > 0.5
        kind = CLEAN if on_target else DIRTY

        owner = pool.step(dets, idx)

        # ціль серед детекцій? (оракульна прив'язка, лише для діагностики)
        gt_det, gt_det_iou = None, 0.5
        for tid, di in owner.items():
            v = iou_xyxy(dets[di], gt_xyxy)
            if not np.isnan(v) and v > gt_det_iou:
                gt_det_iou, gt_det = v, tid

        # бокс, який трекер ТРИМАЄ (для користі)
        held, held_iou = None, 0.5
        for tid, di in owner.items():
            v = iou_xyxy(dets[di], pred_xyxy)
            if not np.isnan(v) and v > held_iou:
                held_iou, held = v, tid

        by_id = {t.id: t for t in pool.tracks}
        row = dict(frame=idx, iou=float(tr_iou), n_dets=len(dets),
                   gt_det=gt_det if gt_det is not None else -1,
                   held=held if held is not None else -1)
        for variant in ('proxy', 'oracle'):
            s = by_id[gt_det].stamp[variant] if gt_det is not None else None
            row[f'tax_{variant}'] = 1 if s else 0
            row[f'tax_{variant}_kind'] = s[1] if s else ''
            row[f'tax_{variant}_at'] = s[0] if s else -1
            h = by_id[held].stamp[variant] if held is not None else None
            row[f'blocked_{variant}'] = 1 if h else 0
        rows.append(row)

        # штампи ставимо ПІСЛЯ зчитування — рішення на кадрі t бачить лише
        # штампи, накопичені до t (причинність)
        stamp_pass(pool, owner, dets, pred_xyxy, idx, kind, 'proxy', args.stamp_min_frames)
        stamp_pass(pool, owner, dets, gt_xyxy, idx, kind, 'oracle', args.stamp_min_frames)

    return rows, pool._next_id


def summarize(video, rows, n_total_tracklets):
    """Зведення по одному відео. Знаменник податку — кадри, де ціль є серед детекцій."""
    have_gt = [r for r in rows if r['gt_det'] >= 0]
    off = [r for r in rows if not np.isnan(r['iou']) and r['iou'] < 0.1 and r['held'] >= 0]
    # ⭐ фрагментація ЦІЛІ: скільки РІЗНИХ tracklet-id по черзі володіли GT-боксом.
    # 1 = ідеальна ідентичність; сотні = ціль розсипана на уламки (корінь отруєння).
    gt_ids = [r['gt_det'] for r in have_gt]
    target_frags = len(set(gt_ids))
    switches = sum(1 for a, b in zip(gt_ids, gt_ids[1:]) if a != b)
    out = dict(video=video, frames=len(rows), gt_visible=len(have_gt), off_target=len(off),
               total_tracklets=n_total_tracklets, target_frags=target_frags,
               target_switches=switches)
    for variant in ('proxy', 'oracle'):
        tax = [r for r in have_gt if r[f'tax_{variant}']]
        clean = [r for r in tax if r[f'tax_{variant}_kind'] == CLEAN]
        out[f'tax_{variant}'] = len(tax) / len(have_gt) if have_gt else np.nan
        out[f'tax_{variant}_clean'] = len(clean) / len(have_gt) if have_gt else np.nan
        out[f'benefit_{variant}'] = (sum(r[f'blocked_{variant}'] for r in off) / len(off)
                                     if off else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='напр. book-19, або повний шлях')
    ap.add_argument('--videos', help='файл зі списком (напр. lasot_probe16.txt)')
    ap.add_argument('--data-dir', default='/home/peoly/datasets/lasot/test')
    ap.add_argument('--tracker', default='YOLOe-VP-IoU')
    ap.add_argument('--tracker-config', default='yoloe-vp-iou/ablation/D0_minimal.yaml',
                    help='шлях БЕЗ префікса configs/ (loader додає сам)')
    ap.add_argument('--model', default='yoloe-v8m-seg.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--link-thr', type=float, default=0.1,
                    help='поріг IoU для зв\'язування tracklet-ів (зворотне дослідження: 0.1)')
    ap.add_argument('--max-age', type=int, default=30, help='кадрів без детекції до смерті tracklet')
    ap.add_argument('--stamp-min-frames', type=int, default=1,
                    help='кадрів роз\'єднаного співіснування до штампа «не ціль» '
                         '(1 = початкова слабка форма; book-19 має запас 962)')
    ap.add_argument('--num-frames', type=int, default=0)
    ap.add_argument('--predict', action='store_true',
                    help='SORT-lite: матчити проти боксу, передбаченого за постійною '
                         'швидкістю (проти фрагментації через зсув під час пропуску)')
    ap.add_argument('--csv-dir', help='каталог для покадрових CSV')
    ap.add_argument('--summary', help='шлях до зведеного CSV')
    args = ap.parse_args()

    if args.videos:
        videos = [ln.split('#')[0].strip() for ln in Path(args.videos).read_text().splitlines()]
        videos = [v for v in videos if v]
    elif args.video:
        videos = [args.video]
    else:
        raise SystemExit('❌ Потрібен --video або --videos')

    print(f'🔗 tracklet: IoU>={args.link_thr}, max_age={args.max_age}, '
          f'штамп після {args.stamp_min_frames} кадр(ів) доказу, '
          f'предикція={"SORT-lite" if args.predict else "off (IoU-only)"} | {len(videos)} відео')
    summaries = []
    for v in videos:
        rows, n_tracklets = run_video(args, v)
        s = summarize(v, rows, n_tracklets)
        summaries.append(s)
        print(f"  {v:<16} уламків_цілі={s['target_frags']:<4} перемикань={s['target_switches']:<4} "
              f"всього_tracklet={s['total_tracklets']:<5} | "
              f"податок_oracle_clean={s['tax_oracle_clean']:.3f}")
        if args.csv_dir:
            d = Path(args.csv_dir)
            d.mkdir(parents=True, exist_ok=True)
            cols = list(rows[0].keys())
            lines = [','.join(cols)]
            lines += [','.join(str(r[c]) for c in cols) for r in rows]
            (d / f'{v}.csv').write_text('\n'.join(lines) + '\n')

    if args.summary:
        cols = list(summaries[0].keys())
        lines = [','.join(cols)]
        lines += [','.join(f'{s[c]:.6f}' if isinstance(s[c], float) else str(s[c])
                           for c in cols) for s in summaries]
        Path(args.summary).write_text('\n'.join(lines) + '\n')
        print(f'💾 {args.summary}')


if __name__ == '__main__':
    main()
