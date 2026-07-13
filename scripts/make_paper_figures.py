#!/usr/bin/env python3
"""
Фігури для статті. Палітра — референсна з dataviz-скіла, перевірена валідатором
(blue #2a78d6 ↔ red #e34948, worst adjacent ΔE 74.6 protan).

Fig 1  Парний scatter на повних 280 відео LaSOT: B9 проти переможця.
       Робота даних — полярність (де краще / де гірше), тому діверджентна пара.
Fig 2  Профіль якості на GOT-10k val: посортований per-video AO.
       Робота даних — розподіл; головне повідомлення — бімодальність.

Обидві фігури — на ПОВНИХ наборах. Числа з 12-відео підвибірки у статтю не йдуть.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── палітра (референсна, світлий фон) ───────────────────────────────
BLUE, RED = '#2a78d6', '#e34948'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
NEUTRAL = '#c9c8c2'          # |Δ| нижче порога помітності

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.labelcolor': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load(path):
    return {r['video_name']: r for r in json.load(open(path))}


def fig1_paired_scatter(out):
    b9 = load('results_ablation_b9/results_B9_full.json/results_YOLOe-VP-IoU.json')
    win = load('results_b10m_floor08_full/results_YOLOe-VP-IoU.json')
    keys = sorted(b9)
    x = np.array([b9[k]['auc'] for k in keys])
    y = np.array([win[k]['auc'] for k in keys])
    d = y - x
    thr = 0.05                      # нижче — не відрізняємо від шуму

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.2, zorder=1)

    groups = [(d > thr, BLUE, f'покращено ({int((d > thr).sum())})'),
              (d < -thr, RED, f'погіршено ({int((d < -thr).sum())})'),
              (np.abs(d) <= thr, NEUTRAL, f'без змін ({int((np.abs(d) <= thr).sum())})')]
    for mask, color, label in groups:
        ax.scatter(x[mask], y[mask], s=26, c=color, alpha=0.80,
                   linewidths=0.6, edgecolors=SURFACE, label=label, zorder=3)

    for k, dx, dy in [('crab-18', 10, -14), ('robot-8', 12, -6), ('chameleon-20', 12, -4),
                      ('pig-10', -18, 12), ('cup-7', 14, -8), ('tank-9', 12, -10)]:
        if k not in b9:
            continue
        ax.annotate(k, (b9[k]['auc'], win[k]['auc']), textcoords='offset points',
                    xytext=(dx, dy), fontsize=7.5, color=INK2,
                    arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.6))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('B9 baseline — AUC')
    ax.set_ylabel('YOLOe-VP-IoU (v8m, floor −0.80) — AUC')
    ax.set_title('Приріст нерівномірний: 83 виграші, 31 програш\n'
                 f'LaSOT, усі 280 відео · середнє {x.mean():.3f} → {y.mean():.3f}',
                 loc='left', color=INK, pad=10)
    leg = ax.legend(loc='upper left', frameon=False, fontsize=8.2, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print('  ✓', out)


def fig2_got10k_profile(out):
    """Розподіл per-video AO. Не бімодальний — сильно лівоасиметричний,
    з окремим хвостом повних відмов. Тому гістограма, а не посортована крива:
    крива показала б плато й обрив, і читач би домалював собі два режими."""
    r = [x for x in json.load(open('results_d0_got10k/results_YOLOe-VP-IoU.json'))
         if x['status'] == 'success']
    ao = np.array([x['ao'] for x in r])
    n = len(ao)
    mean, med = ao.mean(), np.median(ao)
    fail = 0.3                              # поріг «ціль втрачено»

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.grid(True, axis='y', color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    counts, edges = np.histogram(ao, bins=20, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [RED if c < fail else BLUE for c in centers]
    ax.bar(centers, counts, width=0.046, color=colors, linewidth=0, zorder=3)

    ax.axvline(mean, color=INK, linewidth=1.4, linestyle='--', zorder=4)
    ax.axvline(med, color=MUTED, linewidth=1.2, linestyle=':', zorder=4)
    top = counts.max()
    ax.annotate(f'середнє\n{mean:.3f}', (mean, top * 0.92), ha='right', va='top',
                fontsize=8, color=INK, xytext=(-4, 0), textcoords='offset points')
    ax.annotate(f'медіана\n{med:.3f}', (med, top * 0.92), ha='left', va='top',
                fontsize=8, color=INK2, xytext=(4, 0), textcoords='offset points')

    n_lo = int((ao < fail).sum())
    ax.annotate(f'{n_lo} повних відмов\n(AO < 0.3)', (0.02, top * 0.62), ha='left',
                fontsize=8.2, color=RED)
    # свідомо без підпису для синьої частини: він лягав би поверх стовпчиків
    # тим самим кольором. Червоний хвіст — і є повідомлення фігури.

    ax.set_xlim(0, 1); ax.set_ylim(0, top * 1.12)
    ax.set_xlabel('AO (average overlap)')
    ax.set_ylabel('к-ть послідовностей')
    ax.set_title('Хвіст повних відмов тягне середнє на 0.12 нижче за медіану\n'
                 f'D0 на GOT-10k val, {n} послідовностей · AO {mean:.3f}, SR@0.5 0.730',
                 loc='left', color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print('  ✓', out)


if __name__ == '__main__':
    print('фігури:')
    fig1_paired_scatter('figures/fig1_paired_scatter_lasot280.png')
    fig2_got10k_profile('figures/fig2_got10k_profile.png')
