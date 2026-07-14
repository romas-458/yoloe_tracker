#!/usr/bin/env python3
"""
Мінімальна підмножина LaSOT для внутрішньої абляції з МАКСИМАЛЬНИМ покриттям
сценаріїв (задача max-coverage / set-cover по реконструйованих атрибутах).

НЕ плутати з lasot_ablation_subset12 (той — незміщена оцінка СЕРЕДНЬОГО).
Тут ціль інша: кожен тип складності має ≥K семплів, аби абляція бачила ефект
по кожному сценарію, а не ранжувала на шумі.

Атрибути реконструюються з GT (офіційних міток LaSOT не використовуємо):
  FOC — з full_occlusion.txt (частка повністю-оклюдованих кадрів)
  OV  — з out_of_view.txt
  LR  — малий об'єкт: медіана sqrt(area) нижче порога
  ARC — зміна пропорцій: max/min (w/h) > 2 (визначення LaSOT)
  SV  — зміна масштабу: max/min sqrt(area) > 2 (визначення LaSOT)
  FM  — швидкий рух: є кадри, де зсув центру > sqrt(area) (визначення LaSOT)
  LONG/SHORT — за довжиною послідовності
Фотометричні (IV, BC, MB, CM, ROT, DEF, VC, POC) з GT НЕ відновлювані — пропущені.

Складність — з per-video AUC прогону D6b-280 (тір easy/med/hard/vhard).
Інформативний пул: відкидаємо AUC<0.05 (сліпі) та >0.90 (насичені) — там абляція нічого не показує.

Використання:
  python scripts/build_ablation_cover_subset.py [--k 3] [--out lasot_ablation_cover.txt]
"""
import argparse, json, os, glob
import numpy as np

LASOT = "/home/peoly/datasets/lasot/test"
D6B = "results_d6b_lasot280/results_YOLOe-VP-IoU.json"

# --- реконструкція атрибутів одного відео ---------------------------------
def seq_features(seqdir):
    gt = np.loadtxt(os.path.join(seqdir, "groundtruth.txt"), delimiter=",", ndmin=2)
    n = len(gt)
    def load_flag(name):
        p = os.path.join(seqdir, name)
        if not os.path.exists(p):
            return np.zeros(n)
        v = np.loadtxt(p, delimiter=",", ndmin=1)
        return v[:n] if len(v) >= n else np.pad(v, (0, n - len(v)))
    focv = load_flag("full_occlusion.txt")
    ovv = load_flag("out_of_view.txt")
    w, h = gt[:, 2], gt[:, 3]
    cx, cy = gt[:, 0] + w / 2, gt[:, 1] + h / 2
    area = w * h
    valid = (w > 0) & (h > 0) & (focv < 0.5) & (ovv < 0.5)  # видимі кадри з реальним боксом
    vw, vh, va = w[valid], h[valid], area[valid]
    f = {}
    f["FOC"] = focv.mean() > 0.02            # ≥2% кадрів повної оклюзії
    f["OV"] = ovv.mean() > 0.02
    if va.sum() > 0 and len(va) > 5:
        s = np.sqrt(va)
        f["LR"] = np.median(s) < 45          # малий об'єкт (~ <45px сторона)
        f["SV"] = (s.max() / max(s.min(), 1e-6)) > 2.0
        ar = vw / np.maximum(vh, 1e-6)
        f["ARC"] = (ar.max() / max(ar.min(), 1e-6)) > 2.0
    else:
        f["LR"] = f["SV"] = f["ARC"] = False
    # FM: зсув центру між сусідніми кадрами > розмір об'єкта
    d = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    sz = np.sqrt(np.maximum(area[:-1], 0))
    fm_frac = np.mean((d > sz) & (sz > 0)) if len(d) else 0
    f["FM"] = fm_frac > 0.01
    f["LONG"] = n > 3500
    f["SHORT"] = n < 1200
    return f, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="мінімум відео на атрибут")
    ap.add_argument("--k-rare", type=int, default=2, help="мінімум для рідкісних (OV,FOC)")
    ap.add_argument("--tiers", default="2,3,3,2", help="квоти easy,med,hard,vhard")
    ap.add_argument("--out", default="lasot_ablation_cover.txt")
    args = ap.parse_args()

    auc = {os.path.basename(r["video_name"].replace("lasot_test_", "")): r["auc"]
           for r in json.load(open(D6B))}

    seqs = sorted(glob.glob(os.path.join(LASOT, "*", "*")))
    rows = []
    for sd in seqs:
        name = os.path.basename(sd)
        if not os.path.isdir(sd) or not os.path.exists(os.path.join(sd, "groundtruth.txt")):
            continue
        f, n = seq_features(sd)
        a = auc.get(name)
        rows.append({"name": name, "n": n, "auc": a, "attrs": {k for k, v in f.items() if v}})

    ALL_ATTRS = ["FOC", "OV", "LR", "ARC", "SV", "FM", "LONG", "SHORT"]
    # ARC/SV майже універсальні (>90% відео) → недискримінативні, НЕ ведуть відбір
    TARGET_ATTRS = ["FOC", "OV", "LR", "FM", "LONG", "SHORT"]
    RARE = {"OV", "FOC"}
    print(f"Реконструйовано {len(rows)} відео. Поширеність атрибутів (з 280):")
    for at in ALL_ATTRS:
        mark = "  (універсальний, не цільовий)" if at not in TARGET_ATTRS else ""
        print(f"  {at:6} {sum(at in r['attrs'] for r in rows):3d}{mark}")

    # тір складності
    def tier(x):
        if x is None: return "?"
        return "easy" if x >= 0.65 else "med" if x >= 0.45 else "hard" if x >= 0.25 else "vhard"
    for r in rows:
        r["tier"] = tier(r["auc"])

    # інформативний пул: без сліпих/насичених
    pool = [r for r in rows if r["auc"] is not None and 0.05 <= r["auc"] <= 0.90]

    # Клітинки покриття = цільові атрибути + тіри складності (з квотами).
    # Тіри як явні цілі: easy теж, щоб ловити РЕГРЕСІЇ (компонент не має ламати легке).
    _tq = [int(x) for x in args.tiers.split(",")]
    TIER_TARGET = {"easy": _tq[0], "med": _tq[1], "hard": _tq[2], "vhard": _tq[3]}
    att_target = {at: (args.k_rare if at in RARE else args.k) for at in TARGET_ATTRS}
    cover = {at: 0 for at in TARGET_ATTRS}
    tiers = {t: 0 for t in TIER_TARGET}
    chosen = []
    def need():
        return (any(cover[a] < att_target[a] for a in TARGET_ATTRS) or
                any(tiers[t] < TIER_TARGET[t] for t in TIER_TARGET))
    while need() and pool:
        def gain(r):
            g = sum(1 for a in r["attrs"] if a in cover and cover[a] < att_target[a])
            if r["tier"] in tiers and tiers[r["tier"]] < TIER_TARGET[r["tier"]]:
                g += 1  # клітинка тіру
            return g
        pool.sort(key=lambda r: (-gain(r), abs((r["auc"] or 0.5) - 0.45)))
        best = pool.pop(0)
        if gain(best) <= 0:
            break
        chosen.append(best)
        for a in best["attrs"]:
            if a in cover: cover[a] += 1
        tiers[best["tier"]] += 1

    print(f"\nОбрано {len(chosen)} відео.")
    print("Покриття атрибутів: " + " ".join(f"{a}={cover[a]}/{att_target[a]}" for a in TARGET_ATTRS))
    print("Покриття тірів:     " + " ".join(f"{t}={tiers[t]}/{TIER_TARGET[t]}" for t in TIER_TARGET))
    print()
    lines = [f"# LaSOT ablation COVER subset — {len(chosen)} послідовностей (max-coverage set-cover)",
             f"# ціль: ≥{args.k}/атрибут (≥{args.k_rare} рідкісні), баланс складності, інформативний пул (0.05≤AUC≤0.90)",
             "# АТРИБУТИ РЕКОНСТРУЙОВАНІ з GT (FOC,OV,LR,ARC,SV,FM,LONG,SHORT); фотометричні недоступні",
             "#"]
    for r in sorted(chosen, key=lambda r: -(r["auc"] or 0)):
        at = ",".join(sorted(r["attrs"]))
        lines.append(f"{r['name']:<18}# AUC {r['auc']:.3f} {r['tier']:5} n={r['n']:<5} [{at}]")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"✔ записано {args.out}")

if __name__ == "__main__":
    main()
