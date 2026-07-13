#!/usr/bin/env python3
"""
Фігура: як працює мульти-VPE промпт (GOT-10k).

Верхня панель — дві траси per-frame IoU на ОДНОМУ відео:
  single-VPE (один усереднений вектор) vs multi-VPE (anchor/LT/ST як K класів).
Нижня панель — розходження видів пам'яті: мінімальна попарна косинусна схожість
  між активними видами (падає = вигляд цілі змінився) і число активних видів K.

Теза: коли види пам'яті розходяться (косинус падає), єдиний усереднений вектор
розмивається між старим і новим виглядом — і саме там multi-траса тримає ціль,
а single провалюється.

Дані:
  - per-frame IoU: curves_*.npz з прогонів single та multi (--dump-per-frame)
  - розходження видів: рядки MVPE_VIEWS з verbose-прогону multi

Використання:
  python scripts/fig_multi_vpe.py <video> --single-npz A.npz --multi-npz B.npz \
         --views mvpe_log.txt [--out figures/fig4_multi_vpe.png]
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, RED, AQUA = '#2a78d6', '#e34948', '#0f9068'
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


def load_iou(npz_path, video):
    z = np.load(npz_path)
    key = f'{video}|ious'
    if key not in z:
        raise SystemExit(f'{key} немає у {npz_path} — прогнати з --dump-per-frame')
    return z[key]


def load_views(path):
    frames, K, mincos = [], [], []
    for line in open(path):
        if not line.startswith('MVPE_VIEWS'):
            continue
        d = dict(FIELD.findall(line))
        cos = [float(v) for k, v in d.items() if k.startswith('cos_')]
        frames.append(int(d['frame']))
        K.append(int(d['K']))
        mincos.append(min(cos) if cos else 1.0)   # K=1 → немає пар → 1.0
    return np.array(frames), np.array(K), np.array(mincos)


def make(video, single, multi, vf, vK, vcos, out):
    n = min(len(single), len(multi))
    single, multi = single[:n], multi[:n]
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1]})

    # --- верхня панель: дві траси IoU ---
    ax1.grid(True, color=GRID, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.plot(x, single, color=RED, linewidth=1.6, label='single-VPE (один вектор)')
    ax1.plot(x, multi, color=BLUE, linewidth=1.8, label='multi-VPE (anchor/LT/ST)')
    # заливка переваги multi
    ax1.fill_between(x, single, multi, where=multi >= single, color=BLUE, alpha=0.10)
    ax1.fill_between(x, single, multi, where=multi < single, color=RED, alpha=0.10)
    ax1.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=':')

    # позначити кадр, де single остаточно втрачає ціль (IoU→~0 і не відновлюється)
    lost = np.where(single < 0.1)[0]
    collapse = None
    for i in lost:
        if np.all(single[i:] < 0.15):
            collapse = i
            break
    if collapse is not None:
        for ax in (ax1,):
            ax.axvline(collapse, color=MUTED, linewidth=1.0, linestyle='--')
        ax1.annotate('single остаточно\nвтрачає ціль', (collapse, 0.30),
                     xytext=(collapse + 6, 0.42), fontsize=8, color=RED,
                     arrowprops=dict(arrowstyle='->', color=RED, linewidth=0.8))
        ax1.annotate('multi відновлюється\nі тримає до кінця', (collapse + 4, multi[min(collapse + 4, n - 1)]),
                     xytext=(collapse + 10, 0.62), fontsize=8, color=BLUE,
                     arrowprops=dict(arrowstyle='->', color=BLUE, linewidth=0.8))

    ax1.set_ylim(0, 1); ax1.set_ylabel('IoU з ціллю')
    leg = ax1.legend(loc='lower left', frameon=False, fontsize=8.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    d_auc = single.mean(), multi.mean()
    ax1.set_title('Мульти-VPE тримає ціль там, де усереднений вектор її губить\n'
                  f'{video} · середній IoU: single {d_auc[0]:.3f} → multi {d_auc[1]:.3f}',
                  loc='left', color=INK, pad=8)

    # --- нижня панель: розходження видів пам'яті ---
    ax2.grid(True, color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    if len(vf):
        ax2.plot(vf, vcos, color=AQUA, linewidth=1.6)
        ax2.fill_between(vf, 1.0, vcos, color=AQUA, alpha=0.12)
    if collapse is not None:
        ax2.axvline(collapse, color=MUTED, linewidth=1.0, linestyle='--')
    ax2.set_ylabel('схожість\nвидів (min cos)', color=AQUA)
    ax2.tick_params(axis='y', colors=AQUA)
    ax2.set_ylim(min(0.9, (vcos.min() - 0.02) if len(vcos) else 0.9), 1.001)
    ax2.set_xlabel('кадр')

    # K активних видів на другій осі
    axK = ax2.twinx()
    if len(vf):
        axK.step(vf, vK, color=MUTED, linewidth=1.0, where='post')
    axK.set_ylabel('K видів', color=MUTED)
    axK.tick_params(axis='y', colors=MUTED)
    axK.set_ylim(0.5, 3.5); axK.set_yticks([1, 2, 3])
    axK.spines['top'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print('  ✓', out, f'(n={n}, ΔIoU {d_auc[1]-d_auc[0]:+.3f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--single-npz', required=True)
    ap.add_argument('--multi-npz', required=True)
    ap.add_argument('--views', required=True)
    ap.add_argument('--out', default='figures/fig4_multi_vpe.png')
    a = ap.parse_args()
    single = load_iou(a.single_npz, a.video)
    multi = load_iou(a.multi_npz, a.video)
    vf, vK, vcos = load_views(a.views)
    make(a.video, single, multi, vf, vK, vcos, a.out)
