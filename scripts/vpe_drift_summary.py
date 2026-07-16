#!/usr/bin/env python3
"""
Зведення vpe_drift_study.py по набору відео (probe16).

Питання, на яке відповідає: чи є відео, де anchor описує СПРАВЖНІЙ поточний
вигляд цілі краще за ST/agg? Якщо ні — підняття dual_anchor_weight не має
механізму, яким могло б допомогти, і вісь можна закрити без сітки прогонів.

Для кожного відео рахує середній cos(вид, VPE з GT-боксу) окремо в двох режимах:
  на цілі  — кадри з IoU > 0.5
  загублена — кадри з IoU < 0.1
Режим із <20 кадрів вважається невимірним (n/a) — на одному кадрі не рангуємо.

Використання:
  python scripts/vpe_drift_summary.py [--csv-dir results_drift]
                                      [--list lasot_probe16.txt]
                                      [--out figures/vpe_drift_summary.png]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

VIEWS = [('anchor', BLUE), ('LT', AQUA), ('ST', RED), ('agg', PLUM)]
MIN_FRAMES = 20


def read_list(path: Path):
    """lasot_probe16.txt → [(video, half)], half = rescue|hijack за коментарем-заголовком."""
    out, half = [], 'rescue'
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith('#'):
            if 'ПРОГРАЄ' in s:
                half = 'hijack'
            elif 'ВИГРАЄ' in s:
                half = 'rescue'
            continue
        if s:
            out.append((s.split()[0], half))
    return out


def regime_means(df: pd.DataFrame):
    """→ {(режим, вид): середній cos}, режим із замало кадрів → nan."""
    res = {}
    for regime, sel in (('on', df[df.iou > 0.5]), ('lost', df[df.iou < 0.1])):
        for k, _ in VIEWS:
            res[(regime, k)] = sel[k].mean() if len(sel) >= MIN_FRAMES else np.nan
        res[(regime, 'n')] = len(sel)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', default='results_drift')
    ap.add_argument('--list', default='lasot_probe16.txt')
    ap.add_argument('--out', default='figures/vpe_drift_summary.png')
    args = ap.parse_args()

    rows = []
    for video, half in read_list(Path(args.list)):
        f = Path(args.csv_dir) / f'{video}.csv'
        if not f.exists():
            print(f'   ⚠️  нема {f}')
            continue
        m = regime_means(pd.read_csv(f))
        rows.append(dict(video=video, half=half, **{f'{r}_{k}': v for (r, k), v in m.items()}))

    if not rows:
        raise SystemExit('❌ Жодного CSV')
    d = pd.DataFrame(rows)

    # --- Таблиця: хто найкращий опис поточного вигляду, по режимах
    for regime, title in (('on', 'НА ЦІЛІ (IoU>0.5)'), ('lost', 'ЗАГУБЛЕНА (IoU<0.1)')):
        cols = [f'{regime}_{k}' for k, _ in VIEWS]
        sub = d.dropna(subset=cols)
        print(f'\n=== {title} — {len(sub)}/{len(d)} відео вимірні (≥{MIN_FRAMES} кадрів)')
        if not len(sub):
            continue
        print(f'{"відео":<16}{"half":<8}' + ''.join(f'{k:>9}' for k, _ in VIEWS) + f'{"кращий":>9}')
        for _, r in sub.iterrows():
            best = max(VIEWS, key=lambda kv: r[f'{regime}_{kv[0]}'])[0]
            print(f'{r.video:<16}{r.half:<8}'
                  + ''.join(f'{r[f"{regime}_{k}"]:>9.3f}' for k, _ in VIEWS)
                  + f'{best:>9}')
        print(f'{"СЕРЕДНЄ":<24}' + ''.join(f'{sub[f"{regime}_{k}"].mean():>9.3f}' for k, _ in VIEWS))
        wins = {k: sum(max(VIEWS, key=lambda kv: r[f'{regime}_{kv[0]}'])[0] == k
                       for _, r in sub.iterrows()) for k, _ in VIEWS}
        print(f'   перемог як найкращий вид: ' + '  '.join(f'{k}={v}' for k, v in wins.items()))
        n_anchor_beats_agg = int((sub[f'{regime}_anchor'] > sub[f'{regime}_agg']).sum())
        print(f'   anchor > agg: {n_anchor_beats_agg}/{len(sub)} відео '
              f'← якщо 0, підняття anchor-ваги не має механізму допомогти')

    # --- Фігура: anchor − ST на відео, згруповані за половиною probe16
    # sharey=False: панелі містять різні набори відео й сортуються незалежно,
    # тож спільна вісь Y підписала б бари чужими іменами.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (regime, title) in zip(axes, (('on', 'на цілі (IoU>0.5)'),
                                          ('lost', 'загублена (IoU<0.1)'))):
        sub = d.dropna(subset=[f'{regime}_anchor', f'{regime}_ST']).copy()
        if not len(sub):
            ax.set_title(f'{title} — нема даних', loc='left')
            continue
        sub['delta'] = sub[f'{regime}_anchor'] - sub[f'{regime}_ST']
        sub = sub.sort_values('delta')
        colors = [BLUE if h == 'rescue' else RED for h in sub.half]
        ax.barh(range(len(sub)), sub.delta, color=colors, alpha=0.85, zorder=3)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.video, fontsize=8)
        ax.axvline(0, color=INK2, lw=0.9, zorder=4)
        ax.set_xlabel('cos(anchor, GT) − cos(ST, GT)')
        ax.set_title(title, loc='left')
        ax.grid(True, axis='x', color=GRID, lw=0.6, zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.85)]
    axes[0].legend(handles, ['rescue-половина', 'hijack-половина'],
                   frameon=False, fontsize=8.5, loc='lower right')
    fig.suptitle('Чи описує anchor поточний вигляд цілі краще за ST?  '
                 '(праворуч від 0 = так)', x=0.02, ha='left', fontsize=10.5)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'\n🖼  {out}')


if __name__ == '__main__':
    main()
