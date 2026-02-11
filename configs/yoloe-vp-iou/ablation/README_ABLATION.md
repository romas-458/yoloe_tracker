# YOLOe-VP-IoU Ablation Study

## 📋 Мета досліджень

Визначити вплив кожного компонента YOLOe-VP-IoU трекера на загальну продуктивність.

## 🔬 Компоненти для абляції

1. **VPE (Visual Prompt Engineering)** - збір та використання візуальних промптів
2. **Kalman Filter** - згладжування траєкторії та прогнозування
3. **DIoU Metric** - Distance-IoU замість стандартного IoU
4. **Adaptive Confidence** - адаптивний поріг впевненості детекції
5. **Phase 2 DIoU Search** - пошук з DIoU у Phase 2
6. **Phase 3 Validation Gate** - validation після реініціалізації
7. **Dual Memory VPE** - подвійна пам'ять (short-term + long-term)
8. **Warmup Period** - початковий період збору VPE

## 🧪 Експерименти

### Експеримент 1: Baseline (A1)
**Мета**: Встановити базову лінію без додаткових features
- ❌ VPE вимкнено (max_vpe=0)
- ❌ Kalman вимкнено
- ✅ Тільки IoU matching
- ✅ Фіксований confidence

### Експеримент 2: +VPE (A2)
**Мета**: Вплив Visual Prompt Engineering
- ✅ VPE увімкнено (max_vpe=50)
- ❌ Kalman вимкнено
- Порівняти з A1

### Експеримент 3: +Kalman (A3)
**Мета**: Вплив Kalman фільтра без VPE
- ❌ VPE вимкнено
- ✅ Kalman увімкнено
- Порівняти з A1

### Експеримент 4: +VPE+Kalman (A4)
**Мета**: Синергія VPE та Kalman
- ✅ VPE увімкнено
- ✅ Kalman увімкнено
- Порівняти з A2, A3

### Експеримент 5: +DIoU (A5)
**Мета**: Вплив DIoU метрики
- ✅ VPE увімкнено
- ✅ Kalman увімкнено
- ✅ DIoU для Phase 2 та reinit
- Порівняти з A4

### Експеримент 6: +Adaptive (A6)
**Мета**: Вплив адаптивного confidence
- ✅ VPE + Kalman + DIoU
- ✅ Adaptive confidence threshold
- Порівняти з A5

### Експеримент 7: +DualMemory (A7)
**Мета**: Вплив подвійної пам'яті VPE
- ✅ VPE + Kalman + DIoU + Adaptive
- ✅ Dual Memory VPE
- Порівняти з A6

### Експеримент 8: Full (A8)
**Мета**: Повна конфігурація з усіма features
- ✅ Всі компоненти увімкнені
- ✅ Phase 3 Validation
- ✅ Оптимізовані параметри
- Порівняти з A7

## 📈 Очікувані результати

| Експеримент | AUC | Precision | Success@0.5 | FPS | Примітка |
|------------|-----|-----------|-------------|-----|----------|
| A1 Baseline | - | - | - | ~високий | Найшвидший |
| A2 +VPE | ↑ | ↑ | ↑ | ↓ | Робастність ↑ |
| A3 +Kalman | ↑ | ↑ | → | → | Smoothness ↑ |
| A4 +VPE+Kalman | ↑↑ | ↑↑ | ↑ | ↓ | Синергія |
| A5 +DIoU | ↑ | ↑ | ↑↑ | → | Recovery ↑ |
| A6 +Adaptive | ↑ | ↑ | ↑ | → | Precision ↑ |
| A7 +DualMemory | ↑ | ↑ | ↑ | ↓ | Long-term ↑ |
| A8 Full | ↑↑↑ | ↑↑↑ | ↑↑↑ | ↓↓ | Best quality |

## 🚀 Запуск експериментів

### Варіант 1: Автоматичний запуск (рекомендовано)

```bash
# Запустити всі експерименти одним скриптом
./scripts/run_ablation_study.sh
```

Цей скрипт:
- Запускає всі 8 експериментів (A1-A8)
- Використовує тестовий набір з 10 відео
- Зберігає результати у `results_ablation/`
- Генерує відео візуалізацій
- Автоматично створює порівняльну таблицю

### Варіант 2: Ручний запуск окремих експериментів

```bash
# Baseline
python scripts/modular_evaluation.py \
    --dataset-path ~/Datasets/LaSOT/LaSOTTest \
    --output-path results_ablation/results_A1.json \
    --tracker-config configs/yoloe-vp-iou/ablation/A1_baseline.yaml \
    --test-list lasot_ablation_test_list.txt \
    --visualize

# +VPE
python scripts/modular_evaluation.py \
    --dataset-path ~/Datasets/LaSOT/LaSOTTest \
    --output-path results_ablation/results_A2.json \
    --tracker-config configs/yoloe-vp-iou/ablation/A2_vpe.yaml \
    --test-list lasot_ablation_test_list.txt \
    --visualize

# ... та інші експерименти
```

### Варіант 3: Запуск без візуалізації (швидше)

```bash
for config in configs/yoloe-vp-iou/ablation/A*.yaml; do
    exp_name=$(basename "$config" .yaml)
    python scripts/modular_evaluation.py \
        --dataset-path ~/Datasets/LaSOT/LaSOTTest \
        --output-path results_ablation/results_${exp_name}.json \
        --tracker-config "$config" \
        --test-list lasot_ablation_test_list.txt
done
```

## 📊 Аналіз результатів

### Автоматичне порівняння

```bash
# Згенерувати порівняльну таблицю
python scripts/compare_ablation_results.py \
    --results-dir results_ablation \
    --output results_ablation/ablation_comparison.csv
```

Вихід:
```
Ablation Study Results:
===================================================================================
Experiment  Description            Success  Precision  AvgSuccess  AvgPrecision
A1          Baseline (IoU only)    0.5234   0.6123     0.5100      0.5980
A2          +VPE                   0.5678   0.6456     0.5534      0.6312
A3          +Kalman                0.5456   0.6289     0.5312      0.6145
...
```

### Ручний аналіз

Після завершення всіх експериментів:
1. **Порівняти метрики**: AUC, Precision, Success Rate
2. **Проаналізувати швидкість**: FPS для кожної конфігурації
3. **Визначити найважливіші компоненти**: який feature дає найбільший приріст
4. **Оптимізувати trade-off**: точність vs швидкість
5. **Проаналізувати per-sequence**: які features допомагають на складних відео

### Ключові питання для аналізу

1. **Синергія**: Чи дає A4 (VPE+Kalman) більше, ніж A2+A3 окремо?
2. **DIoU ефект**: Наскільки DIoU покращує recovery (A5 vs A4)?
3. **Adaptive важливість**: Чи виправдовує адаптивний conf складність (A6 vs A5)?
4. **Dual Memory**: Чи допомагає подвійна пам'ять на довгих відео (A7 vs A6)?
5. **Full оптимізація**: Чи Full конфігурація (A8) кращий за incrementальний A7?
