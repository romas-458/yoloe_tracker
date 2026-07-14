#!/usr/bin/env python3
"""
OPE success / normalized-precision plots (конвенція LaSOT/OTB).

Кожна крива — усереднення по всіх послідовностях прогону (рівна вага на відео).
Дані беруться з curves_<tracker>.npz, що їх пише modular_evaluation --dump-per-frame:
ключі `<video>|success` (101), `<video>|precision` (51), `<video>|norm_precision` (51).
Осі (з modular_evaluation._compute_curves):
  success        overlap threshold  np.arange(0, 1.01, 0.01)  → AUC = mean кривої
  precision      location error px  np.arange(0, 51, 1.0)     → P@20 = крива[20]
  norm_precision нормалізована      np.linspace(0, 0.5, 51)   → Pnorm = mean кривої

Легенда сортується за AUC (спадання); колір закріплений за методом (порядок вводу),
НЕ за рангом. Метод "наш" (перший або --ours) виділено товщою лінією поверх решти.

Палітра — референсна з dataviz-скіла (порядок слотів валідований на CVD):
  blue aqua yellow green violet red.

Використання:
  python scripts/fig_ope_plots.py \
      "Single (D0)=results_d0_lasot12/curves_YOLOe-VP-IoU.npz" \
      "Multi (D5)=results_d5_lasot12/curves_YOLOe-VP-IoU.npz" \
      --out figures/fig_ope_lasot12.png --dataset "LaSOT (12 послідовностей)"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── стиль (спільний із make_paper_figures.py) ───────────────────────
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
# категоріальний порядок слотів (референсна палітра, валідована на CVD)
PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948',
           '#e87ba4', '#eb6834']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.labelcolor': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
})

SUCC_X = np.arange(0, 1.01, 0.01)      # 101
PREC_X = np.arange(0, 51, 1.0)         # 51 px
NPREC_X = np.linspace(0, 0.5, 51)      # 51

PANELS = {
    # key: (x, xlabel, ylabel, title, score_fn -> (value, "легенда-суфікс"))
    'success': (SUCC_X, 'Поріг перекриття', 'Success rate', 'Success plot',
                lambda c: (c.mean(), 'AUC')),
    'precision': (PREC_X, 'Похибка локалізації (px)', 'Precision', 'Precision plot',
                  lambda c: (c[20], 'P@20')),
    'norm_precision': (NPREC_X, 'Нормалізована похибка', 'Precision', 'Normalized precision plot',
                       lambda c: (c.mean(), 'Pnorm')),
}


def mean_curve(npz_path, kind):
    """Середня крива по всіх відео у прогоні (рівна вага на відео)."""
    d = np.load(npz_path, allow_pickle=True)
    curves = [d[k] for k in d.files if k.endswith('|' + kind)]
    if not curves:
        raise SystemExit(f"❌ {npz_path}: немає кривих '|{kind}' (запусти з --dump-per-frame)")
    return np.mean(np.stack(curves, axis=0), axis=0), len(curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='LABEL=path/to/curves_*.npz (кілька)')
    ap.add_argument('--panels', default='success,norm_precision',
                    help='через кому: success,precision,norm_precision')
    ap.add_argument('--out', default='figures/fig_ope.png')
    ap.add_argument('--dataset', default='', help='підпис набору для заголовка')
    ap.add_argument('--ours', default=None, help='LABEL, який виділити (типово перший)')
    args = ap.parse_args()

    runs = []  # (label, path, color)
    for i, spec in enumerate(args.runs):
        if '=' not in spec:
            raise SystemExit(f"❌ очікую LABEL=path, отримав: {spec}")
        label, path = spec.split('=', 1)
        runs.append([label.strip(), path.strip(), PALETTE[i % len(PALETTE)]])
    ours = args.ours or runs[0][0]

    panels = [p.strip() for p in args.panels.split(',') if p.strip()]
    for p in panels:
        if p not in PANELS:
            raise SystemExit(f"❌ невідома панель '{p}'. Доступні: {list(PANELS)}")

    n_seq = None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0))
    if len(panels) == 1:
        axes = [axes]

    for ax, pkey in zip(axes, panels):
        x, xlab, ylab, title, score_fn = PANELS[pkey]
        # порахувати всі криві + бали, відсортувати легенду за балом (спадання)
        entries = []
        for label, path, color in runs:
            curve, k = mean_curve(path, pkey)
            n_seq = k
            val, suf = score_fn(curve)
            entries.append((val, label, curve, color, suf))
        entries.sort(key=lambda e: -e[0])

        for val, label, curve, color, suf in entries:
            emph = (label == ours)
            ax.plot(x, curve, color=color,
                    lw=2.6 if emph else 1.8,
                    alpha=1.0 if emph else 0.9,
                    zorder=3 if emph else 2,
                    label=f'{label} [{suf} {val:.3f}]')
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 1)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(title, color=INK)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        leg = ax.legend(loc='lower left' if pkey == 'success' else 'lower right',
                        frameon=False, fontsize=8, handlelength=1.4)
        for t in leg.get_texts():
            t.set_color(INK2)

    ds = f' — {args.dataset}' if args.dataset else ''
    fig.suptitle(f'One-pass evaluation{ds}' + (f' · {n_seq} відео' if n_seq else ''),
                 fontsize=11, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f'✔ {args.out}  ({n_seq} відео, панелі: {", ".join(panels)})')


if __name__ == '__main__':
    main()
