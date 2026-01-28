"""
SAMURAI Integration Utilities

Реалізація функцій для інтеграції SAMURAI Kalman фільтра з трекінгом

✨ Функції:
1. select_best_mask_with_kalman() - Equation 7 (гібридна оцінка)
2. select_memory_bank() - Section 4.2 (вибір референтних кадрів)
3. compute_hybrid_association_scores() - Матриця для асоціації
"""

from typing import List, Tuple, Optional
import numpy as np


def select_best_mask_with_kalman(
    masks: List,
    affinity_scores: np.ndarray,
    kalman_iou_scores: np.ndarray,
    alpha_kf: float = 0.15
) -> Tuple[int, np.ndarray]:
    """
    Вибрати найкращу маску на основі гібридної оцінки (Equation 7 з SAMURAI)

    SAMURAI Equation 7:
    M* = argmax_i(α_kf · s_kf(M_i) + (1-α_kf) · s_mask(M_i))

    Стаття показала, що комбінування motion (s_kf) з appearance (s_mask)
    дає +2-5% поліпшення на GOT-10k (Table 1)

    Args:
        masks: Список масок від SAM2 або детектора
        affinity_scores: np.array форма (N,) - s_mask (якість маски від детектора)
                        значення від 0 до 1
        kalman_iou_scores: np.array форма (N,) - s_kf = IoU(predicted_bbox, mask)
                          значення від 0 до 1
        alpha_kf: float - вага motion score (стаття рекомендує α_kf = 0.15)
                 motion отримує 15%, appearance отримує 85%

    Returns:
        Tuple[best_mask_idx, hybrid_scores]:
            best_mask_idx: int - індекс вибраної маски
            hybrid_scores: np.array - всі гібридні оцінки для отримання інформації

    Приклад:
        >>> masks = [mask1, mask2, mask3]
        >>> affinity = np.array([0.8, 0.9, 0.7])  # SAM2 confidence
        >>> kalman_iou = np.array([0.6, 0.4, 0.9])  # IoU з prediction
        >>>
        >>> best_idx, hybrid_scores = select_best_mask_with_kalman(
        ...     masks, affinity, kalman_iou, alpha_kf=0.15
        ... )
        >>> # hybrid_scores = 0.15*[0.6,0.4,0.9] + 0.85*[0.8,0.9,0.7]
        >>> #              = [0.768, 0.766, 0.770]
        >>> # best_idx = 2

    Результати SAMURAI (Table 4):
        - α_kf = 0.15 показує найкращі результати
        - Покращення: AO +2.9%, OP_0.5 +5.3%, OP_0.75 +1.9%
    """
    affinity_scores = np.asarray(affinity_scores, dtype=np.float32)
    kalman_iou_scores = np.asarray(kalman_iou_scores, dtype=np.float32)

    # Комбінована оцінка (Equation 7)
    hybrid_scores = (
        alpha_kf * kalman_iou_scores +
        (1 - alpha_kf) * affinity_scores
    )

    # Вибрати маску з найвищою гібридною оцінкою
    best_idx = int(np.argmax(hybrid_scores))

    return best_idx, hybrid_scores


def select_memory_bank(
    recent_frames: List[dict],
    motion_scores: np.ndarray,
    affinity_scores: np.ndarray,
    object_confidence_scores: np.ndarray,
    N_max: int = 5,
    selection_threshold: Optional[float] = None
) -> Tuple[List[dict], np.ndarray]:
    """
    Вибрати кадри для пошуку маски на основі motion + affinity + object confidence

    SAMURAI Section 4.2 (Memory Bank Selection):
    B_t = {m_i | f(s_mask, s_obj, s_kf) = 1, t - N_max ≤ i < t}

    Коментар зі статті: "we also maintain a memory bank of frames where
    the tracked object has been consistently present and accurately
    detected to reduce computational load and improve robustness"

    Логіка:
    1. Обчислити комбіновану оцінку для кожного кадру
    2. Вибрати кадри з оцінкою вище порогу
    3. Зберегти останні N_max найкращих кадрів

    Args:
        recent_frames: List[dict] - останні кадри з інформацією
        motion_scores: np.array форма (N,) - якість motion (use_motion flag)
        affinity_scores: np.array форма (N,) - s_mask (маска якість)
        object_confidence_scores: np.array форма (N,) - впевненість детектора
        N_max: int - максимальна довжина пам'яті (рекомендація: 5)
        selection_threshold: Optional[float] - мінімальна оцінка для вибору
                            якщо None, використовувати медіану

    Returns:
        Tuple[selected_frames, selection_mask]:
            selected_frames: List[dict] - вибрані кадри
            selection_mask: np.array bool - маска для вибору

    Приклад:
        >>> recent_frames = [
        ...     {'frame_id': 1, 'bbox': [...], 'mask': [...]},
        ...     {'frame_id': 2, 'bbox': [...], 'mask': [...]},
        ...     ...
        ... ]
        >>> motion_scores = np.array([0.0, 1.0, 1.0, 0.0, 1.0])  # use_motion
        >>> affinity = np.array([0.8, 0.9, 0.7, 0.5, 0.85])  # SAM confidence
        >>> obj_conf = np.array([0.7, 0.8, 0.9, 0.3, 0.85])  # detector conf
        >>>
        >>> selected, mask = select_memory_bank(
        ...     recent_frames, motion_scores, affinity, obj_conf
        ... )
        >>> len(selected)  # <= N_max=5
        >>> selected[0]['frame_id']  # найкращі кадри

    Ваги (рекомендація з SAMURAI):
    - 30% motion якість (use_motion indicator)
    - 40% маска якість (s_mask від SAM2)
    - 30% детектор впевненість (s_obj)
    """
    motion_scores = np.asarray(motion_scores, dtype=np.float32)
    affinity_scores = np.asarray(affinity_scores, dtype=np.float32)
    object_confidence_scores = np.asarray(object_confidence_scores, dtype=np.float32)

    # Нормалізувати оцінки [0, 1] якщо потребується
    if motion_scores.max() > 1.0:
        motion_scores = motion_scores / motion_scores.max()
    if affinity_scores.max() > 1.0:
        affinity_scores = affinity_scores / affinity_scores.max()
    if object_confidence_scores.max() > 1.0:
        object_confidence_scores = object_confidence_scores / object_confidence_scores.max()

    # Комбінована оцінка (рекомендовані ваги)
    combined_scores = (
        0.3 * motion_scores +           # 30% - motion якість
        0.4 * affinity_scores +          # 40% - маска якість
        0.3 * object_confidence_scores   # 30% - детектор впевненість
    )

    # Визначити поріг
    if selection_threshold is None:
        selection_threshold = np.median(combined_scores)

    # Вибрати кадри з оцінкою >= порог
    selection_mask = combined_scores >= selection_threshold

    # Обмежити на N_max найкращих
    selected_indices = np.where(selection_mask)[0]
    if len(selected_indices) > N_max:
        # Вибрати N_max найкращих
        best_indices = np.argsort(combined_scores[selected_indices])[-N_max:]
        selected_indices = selected_indices[best_indices]
        selection_mask = np.zeros_like(selection_mask)
        selection_mask[selected_indices] = True

    # Зберегти порядок кадрів
    selected_frames = [
        f for i, f in enumerate(recent_frames)
        if selection_mask[i]
    ]

    return selected_frames, selection_mask


def compute_hybrid_association_scores(
    kalman_iou_matrix: np.ndarray,
    affinity_scores_matrix: np.ndarray,
    alpha_kf: float = 0.15
) -> np.ndarray:
    """
    Обчислити матрицю гібридних оцінок для асоціації треків до детекцій

    ✨ Equation 7 применено до асоціаційної матриці

    Args:
        kalman_iou_matrix: np.array форма (num_tracks, num_detections)
                          IoU між predicted bbox трека і detected bbox
        affinity_scores_matrix: np.array форма (num_tracks, num_detections)
                               affinity score від детектора
        alpha_kf: float - вага motion score (default: 0.15 з SAMURAI)

    Returns:
        hybrid_scores: np.array форма (num_tracks, num_detections)
                      гібридні оцінки для асоціації

    Приклад:
        >>> kalman_iou = np.array([
        ...     [0.8, 0.2, 0.1],
        ...     [0.1, 0.9, 0.3],
        ...     [0.2, 0.3, 0.7]
        ... ])  # 3 треки, 3 детекції
        >>> affinity = np.array([
        ...     [0.7, 0.3, 0.2],
        ...     [0.2, 0.8, 0.4],
        ...     [0.3, 0.4, 0.9]
        ... ])
        >>>
        >>> hybrid = compute_hybrid_association_scores(kalman_iou, affinity)
        >>> # hybrid = 0.15*kalman_iou + 0.85*affinity
    """
    kalman_iou_matrix = np.asarray(kalman_iou_matrix, dtype=np.float32)
    affinity_scores_matrix = np.asarray(affinity_scores_matrix, dtype=np.float32)

    # Гібридна матриця оцінок
    hybrid_scores = (
        alpha_kf * kalman_iou_matrix +
        (1 - alpha_kf) * affinity_scores_matrix
    )

    return hybrid_scores


def compute_motion_confidence(
    successful_frames: int,
    tau_kf: int = 3,
    decay_rate: float = 0.1
) -> float:
    """
    Обчислити впевненість motion моделі на основі stability gate

    Args:
        successful_frames: int - кількість успішних оновлень
        tau_kf: int - поріг для активації (default: 3)
        decay_rate: float - швидкість занепаду впевненості (default: 0.1)

    Returns:
        float - впевненість від 0 до 1
    """
    if successful_frames < tau_kf:
        # Не активна - впевненість 0
        return 0.0
    elif successful_frames == tau_kf:
        # Щойно активована - впевненість 0.5
        return 0.5
    else:
        # Рівномірно зростає до 1.0
        extra_frames = successful_frames - tau_kf
        confidence = min(1.0, 0.5 + extra_frames * decay_rate)
        return float(confidence)


def compute_iou_stats(iou_scores: np.ndarray) -> dict:
    """
    Обчислити статистику IoU оцінок для діагностики

    Args:
        iou_scores: np.array - IoU оцінки

    Returns:
        dict з статистикою
    """
    iou_scores = np.asarray(iou_scores, dtype=np.float32)
    return {
        'mean': float(np.mean(iou_scores)),
        'std': float(np.std(iou_scores)),
        'max': float(np.max(iou_scores)),
        'min': float(np.min(iou_scores)),
        'median': float(np.median(iou_scores))
    }


# ✨ SAMURAI Constants (як у статті)
SAMURAI_DEFAULTS = {
    'alpha_kf': 0.15,           # Equation 7: motion вага 15%, appearance 85%
    'tau_kf': 3,                # Section 4.1: успішних фреймів для активації
    'N_max': 5,                 # Section 4.2: розмір memory bank
    'motion_weight': 0.3,       # Section 4.2: вага motion у memory selection
    'affinity_weight': 0.4,     # Section 4.2: вага affinity
    'confidence_weight': 0.3,   # Section 4.2: вага object confidence
}

__all__ = [
    'select_best_mask_with_kalman',
    'select_memory_bank',
    'compute_hybrid_association_scores',
    'compute_motion_confidence',
    'compute_iou_stats',
    'SAMURAI_DEFAULTS'
]
