# Візуалізаційні скрипти

Як будувати фігури й анотовані кадри трекера.

Спільне для всіх прогонів трекера:

- усі прогони йдуть через `tracking_dev` python:
  ```bash
  PY=/home/peoly/anaconda3/envs/tracking_dev/bin/python
  ```
- шлях до конфіга в `--tracker-config` передається **без** префікса `configs/`
  (лоадер додає його сам), напр. `yoloe-vp-iou/ablation/D5_multi_vpe.yaml`.
- ⚠️ **`--model` не діє при `--tracker-config`**. Порядок злиття параметрів:
  `--model` (1) → конфіг (2, перезаписує) → `--tracker-params` (3, найвищий
  пріоритет). Тобто `model_path` з yaml завжди б'є `--model`. Щоб змінити модель:
  або візьми відповідний конфіг (напр. `..._v8l.yaml`), або перевизнач через
  ```bash
  --tracker-params '{"model_path": "yoloe-v8l-seg.pt"}'
  ```
  (Примітка: прапорець зветься `--repetitions`, з `s`.)

---

## 1. Fig4 — multi-VPE (усе автоматизовано)

Обгортка проганяє відео двічі (D0 single + D5 multi verbose), робить
`--dump-per-frame` і будує порівняльну фігуру.

```bash
# типово: GOT-10k_Val_000111 → figures/fig4_multi_vpe.png
bash scripts/make_fig_multi_vpe.sh

# своє відео + свій вихід
bash scripts/make_fig_multi_vpe.sh GOT-10k_Val_000028 figures/fig4_val028.png
```

Відео з сильним виграшем multi: `000111`, `000028`, `000025`.
Якщо в логах 0 рядків `MVPE_VIEWS` — multi-VPE не активувався
(перевір `multi_vpe_prompt` / dual memory).

---

## 2. Анотовані кадри (`--visualize`)

Кадри пишуться в `<output>/val/<video>/`; кандидати multi-VPE фарбуються за
видом пам'яті (anchor / LT / ST).

```bash
$PY scripts/modular_evaluation.py \
  -d /home/peoly/datasets/got10k/got_10k_val/val \
  -o results_vis_multivpe \
  --tracker-config yoloe-vp-iou/ablation/D5_multi_vpe.yaml \
  --dataset got10k --video GOT-10k_Val_000111 \
  --visualize --num-frames 100000
```

---

## 3. Fig3 — площина DIoU×conf (joint-бал)

Двокроково: спершу verbose-прогін ловить рядки `JOINTCAND`, тоді скрипт малює.

⚠️ GT-шлях у `fig_joint_plane.py` захардкоджений під **LaSOT test**
(`/home/peoly/datasets/lasot/test/...`) — бери LaSOT-відео з увімкненим
`use_joint_score`.

```bash
# 1) зібрати JOINTCAND (потрібен verbose config з joint-балом, напр. C0_full)
$PY scripts/modular_evaluation.py -d /home/peoly/datasets/lasot/test \
  -o /tmp/jc --tracker-config yoloe-vp-iou/ablation/C0_full.yaml \
  --dataset lasot --video basketball-1 --num-frames 100000 2>&1 \
  | tr '\r' '\n' | grep '^JOINTCAND' > jc.txt

# 2) фігура (без --frame сам обере кадр перехоплення дистрактором)
$PY scripts/fig_joint_plane.py jc.txt basketball-1 --out figures/fig3_joint_plane.png
# конкретний кадр:  --frame 137
```

Конфіг має мати `verbose: true`. Майже всі ablation-конфіги йдуть з
`verbose: false`, тож рядків `JOINTCAND` під ним **не буде** (порожній `jc.txt`) —
спершу зроби verbose-копію, як обгортка з п.1:
```bash
sed 's/^verbose: false/verbose: true/' \
  configs/yoloe-vp-iou/ablation/C0_full.yaml > /tmp/C0_verbose.yaml
```

Якщо авто-вибір кадру падає з `кадру з перехопленням не знайдено` — це нормально
для «чистих» відео, де топ-кандидат за балом завжди і є ціллю (hijack=False).
Тоді задай `--frame N` вручну, взявши кадр із найбільшою к-стю кандидатів:
```bash
grep -oE 'frame=[0-9]+' jc.txt | sort | uniq -c | sort -rn | head
# → напр. для basketball-1 найбагатший кадр 216:
$PY scripts/fig_joint_plane.py jc.txt basketball-1 --frame 216 --out figures/fig3_joint_plane.png
```

---

## 4. Fig1 / Fig2 — агрегатні (без аргументів)

```bash
$PY scripts/make_paper_figures.py
```

Читає **захардкоджені** результати-JSON (gitignored, але присутні локально):

- **fig1** (LaSOT-280 paired scatter):
  `results_ablation_b9/…B9_full…` vs `results_b10m_floor08_full/…`
- **fig2** (GOT-10k профіль AO): `results_d0_got10k/…`

Ці `results_*` теки мають існувати локально — якщо видалено, спершу треба
перегнати відповідні прогони.

---

## 5. Thumbnail архітектури

```bash
$PY scripts/make_arch_thumb.py   # рендерить прев'ю з figures/architecture.html
```
