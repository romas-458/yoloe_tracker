#!/usr/bin/env python3
"""
YOLOe-VP-IoU Tracker

Гібридний трекер що комбінує:
- Visual Prompt (VP) detection з adaptive VPE aggregation
- IoU matching для refined tracking

Переваги:
- VP знаходить кандидатів (robust до appearance змін)
- IoU вибирає найближчий об'єкт (geometric consistency)
- Adaptive VPE збирається для покращення VP якості
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# YOLOe imports
try:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    YOLOE_AVAILABLE = True
except ImportError:
    YOLOE_AVAILABLE = False
    print("⚠️  YOLOE не доступна")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trackers.base_tracker import BaseTracker, register_tracker
from samurai_integration import (
    select_best_mask_with_kalman,
    compute_hybrid_association_scores,
    compute_motion_confidence,
    SAMURAI_DEFAULTS
)
from trackers.dual_memory_vpe import DualMemoryVPE


class KalmanBoxTrackerSimple:
    """
    Простий Калман фільтр для bbox трекінгу без OpenCV (для сумісності з OpenCV 4.12.0+)

    Вектор стану: [x, y, w, h, vx, vy, vw, vh]
    Використовує стандартне матричне множення через NumPy
    """

    def __init__(self, bbox: List[float], process_noise: float = 0.01, measurement_noise: float = 10.0):
        """
        Простий 8D Kalman фільтр через NumPy

        Args:
            bbox: [x, y, w, h]
            process_noise: Шум моделі
            measurement_noise: Шум вимірювання
        """
        x, y, w, h = bbox

        # Стан: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.array([float(x), float(y), float(w), float(h), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Матриця переходу (dt=1)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0  # x' = x + vx
        self.F[1, 5] = 1.0  # y' = y + vy
        self.F[2, 6] = 1.0  # w' = w + vw
        self.F[3, 7] = 1.0  # h' = h + vh

        # Матриця вимірювання
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Коваріація помилки
        self.P = np.eye(8, dtype=np.float64) * 1000.0

        # Шум процесу
        self.Q = np.eye(8, dtype=np.float64) * process_noise

        # Шум вимірювання
        self.R = np.eye(4, dtype=np.float64) * measurement_noise

    def predict(self) -> List[float]:
        """Прогноз"""
        # P = F @ P @ F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        # x = F @ x
        self.state = self.F @ self.state
        return [float(self.state[0]), float(self.state[1]), float(self.state[2]), float(self.state[3])]

    def update(self, bbox: List[float]) -> List[float]:
        """Оновлення з вимірюванням"""
        z = np.array([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], dtype=np.float64)

        # S = H @ P @ H.T + R
        S = self.H @ self.P @ self.H.T + self.R
        # K = P @ H.T @ S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S + 1e-10)
        # y = z - H @ x
        y = z - self.H @ self.state
        # x = x + K @ y
        self.state = self.state + K @ y
        # P = (I - K @ H) @ P
        self.P = (np.eye(8) - K @ self.H) @ self.P

        return [float(self.state[0]), float(self.state[1]), float(self.state[2]), float(self.state[3])]

    def get_state(self) -> List[float]:
        """Отримати поточний стан"""
        return [float(self.state[i]) for i in range(4)]


class KalmanBoxTracker:
    """
    Розширений Калман фільтр для bbox трекінгу з SAMURAI інтеграцією

    Вектор стану: [x, y, w, h, vx, vy, vw, vh]
    Вимірювання: [x, y, w, h]

    ✨ SAMURAI Features:
    - IoU-based motion scoring (Equation 6: s_kf = IoU(x̂_{t+1|t}, M))
    - Stability gate (τ_kf = 3 successful frames before motion activation)
    - Motion-aware score for hybrid tracking (Equation 7: α_kf·s_kf + (1-α_kf)·s_mask)

    Забезпечує:
    - Згладжування траєкторії bbox
    - Прогнозування положення у наступному кадрі
    - Компенсацію шуму детекції
    - IoU scoring для гібридної асоціації
    - Стійкість до оклюзії через stability gate
    """

    def __init__(self, bbox: List[float], process_noise: float = 0.01, measurement_noise: float = 10.0, track_id: Optional[int] = None):
        """
        Ініціалізація розширеного Калман фільтра з SAMURAI інтеграцією

        Args:
            bbox: [x, y, w, h]
            process_noise: Шум процесу (вища = швидша адаптація до змін руху)
            measurement_noise: Шум вимірювання (вища = більше згладжування, менша реакція на шум)
            track_id: ID треку для логування (опціонально)
        """
        # Використовуємо NumPy реалізацію для сумісності з OpenCV 4.12.0+
        self._kf_simple = KalmanBoxTrackerSimple(bbox, process_noise, measurement_noise)
        self.track_id = track_id

        # ✨ SAMURAI Stability Gate (Section 4.1)
        # Активувати motion модель тільки після τ_kf успішних оновлень
        self.successful_frames = 0
        self.tau_kf = 3  # Paper recommendation
        self.use_motion = False

    def predict(self) -> List[float]:
        """
        Прогноз положення bbox у наступному кадрі на основі поточного стану та швидкості

        Returns:
            Прогнозований bbox [x, y, w, h]
        """
        return self._kf_simple.predict()

    def predict_with_iou_scores(self, detected_boxes: List[List[float]]) -> Tuple[List[float], np.ndarray]:
        """
        Передбачити позицію і обчислити IoU з кожним обнаруженим боксом

        ✨ Equation 6 з SAMURAI: s_kf = IoU(x̂_{t+1|t}, M)

        Args:
            detected_boxes: Список [x, y, w, h] - кандидати для оновлення

        Returns:
            Tuple[predicted_box, iou_scores]:
                predicted_box: [x, y, w, h] - передбачена позиція
                iou_scores: np.array з IoU між predicted_box і кожним detected_box
        """
        predicted_box = self.predict()

        iou_scores = []
        for det_box in detected_boxes:
            iou = self._compute_iou(predicted_box, det_box)
            iou_scores.append(float(iou))

        return predicted_box, np.array(iou_scores)

    def update(self, bbox: List[float], is_successful: bool = True) -> List[float]:
        """
        Оновлення стану з новим вимірюванням (детекцією) з логікою stability gate

        ✨ SAMURAI Stability Gate (Section 4.1):
        Включаємо motion модель тільки якщо tracked object успішно оновлювався
        протягом минулих τ_kf кадрів

        Args:
            bbox: Виміряний bbox [x, y, w, h]
            is_successful: Чи було оновлення успішним (залежить від IoU matching результату)

        Returns:
            Згладжений bbox [x, y, w, h]
        """
        if is_successful:
            self.successful_frames += 1
            # Активувати motion модель тільки після τ_kf успішних оновлень
            if self.successful_frames >= self.tau_kf:
                self.use_motion = True
        else:
            # Скинути лічильник при помилці (оклюзія, реідентифікація)
            self.successful_frames = 0
            self.use_motion = False

        return self._kf_simple.update(bbox)

    def get_motion_score(self) -> float:
        """
        Отримати motion score для гібридної оцінки (Equation 7)

        Використовується в гібридній оцінці:
        M* = argmax(α_kf·s_kf + (1-α_kf)·s_mask)

        Returns:
            float: 1.0 якщо use_motion=True (motion модель активна), 0.0 інакше
        """
        return 1.0 if self.use_motion else 0.0

    def get_state(self) -> List[float]:
        """
        Отримати поточний стан без прогнозу

        Returns:
            Поточний bbox [x, y, w, h]
        """
        return self._kf_simple.get_state()

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """
        Обчислити IoU між двома боксами в центро-розмірному форматі

        Args:
            box1: [x_center, y_center, width, height]
            box2: [x_center, y_center, width, height]

        Returns:
            float: IoU від 0 до 1
        """
        # Конвертувати з центро-розмірного формату в corner format
        x1_min = float(box1[0] - box1[2] / 2)
        y1_min = float(box1[1] - box1[3] / 2)
        x1_max = float(box1[0] + box1[2] / 2)
        y1_max = float(box1[1] + box1[3] / 2)

        x2_min = float(box2[0] - box2[2] / 2)
        y2_min = float(box2[1] - box2[3] / 2)
        x2_max = float(box2[0] + box2[2] / 2)
        y2_max = float(box2[1] + box2[3] / 2)

        # Обчислити перетин
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        # Якщо немає перетину
        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        return float(intersection / (union + 1e-6))


@register_tracker
class YOLOeVPIoUTracker(BaseTracker):
    """
    YOLOe Visual Prompt + IoU Hybrid Tracker

    Гібридний трекер з adaptive VPE та IoU matching.

    ВАЖЛИВО: Використовуйте моделі БЕЗ суфікса '-pf' (prompt-free)!
    Підтримувані моделі: yoloe-26s-seg.pt, yoloe-v8l-seg.pt, etc.
    НЕ підтримуються: yoloe-26s-seg-pf.pt (prompt-free моделі)

    Алгоритм:
    1. VP detection знаходить кандидатів того ж "класу" (через VPE)
    2. IoU matching вибирає найближчого до попередньої позиції
    3. VPE збирається кожні vpe_step кадрів для адаптації
    4. Агреговані VPE покращують VP якість

    Parameters:
        model_path: str - шлях до YOLOe моделі (БЕЗ '-pf'!)
        conf: float - мінімальна впевненість detection (default: 0.1)
        imgsz: int - розмір вхідного зображення (default: 640)
        device: str - пристрій ('cuda' або 'cpu', default: 'cuda')
        iou_threshold: float - мінімальний IoU для matching (default: 0.3)
        vpe_step: int - крок збору VPE (default: 10)
        max_vpe: int - максимальна кількість VPE (default: 5)
        max_lost_frames: int - максимум кадрів без matching перед реідентифікацією (default: 30)
        vpe_conf_threshold: float - мінімальна conf для збору VPE (default: 0.5)
        use_kalman: bool - використовувати Калман фільтр для згладжування та прогнозування (default: False)
        kalman_process_noise: float - шум процесу Калмана (вищий = швидша адаптація, default: 0.01)
        kalman_measurement_noise: float - шум вимірювання Калмана (вищий = більше згладжування, default: 10.0)
        phase2_diou_threshold: float - мінімальний DIoU для Phase 2 matching (>=1.0 = disabled, default: 1.0)
            Дозволяє знаходити об'єкт на Phase 2 навіть якщо IoU=0 (bbox не перетинаються)
            Рекомендовані значення: -0.5 (помірне), -0.7 (м'яке), 0.0 (строге)
        waiting_reinit_conf_threshold: float - мінімальна conf для дострокової реініціалізації у Phase 2 (>=1.0 = disabled, default: 1.0)
            Якщо conf >= цього порогу та diou >= waiting_reinit_diou_threshold, то реініціалізація відбувається одразу
        waiting_reinit_diou_threshold: float - мінімальний DIoU для дострокової реініціалізації у Phase 2 (default: -0.5)
            Працює тільки якщо waiting_reinit_conf_threshold < 1.0
            Рекомендовані значення: -0.5 (помірне), -0.3 (строге), -0.7 (м'яке)
        reinit_diou_threshold: float - початковий мінімальний DIoU для реідентифікації (-1.0 = без обмеження, default: -1.0)
            DIoU = IoU - (d²/c²), де d - відстань центрів, c - діагональ охоплюючого bbox
            Діапазон: від -1 (дуже далеко) до 1 (ідеальне співпадіння)
            Рекомендовані значення: -0.5 (слабке обмеження), 0.0 (помірне), 0.3 (сильне)
        reinit_diou_max: float - максимальний (слабкий) DIoU threshold після тривалої втрати (default: -0.9)
            Якщо > reinit_diou_threshold: threshold адаптивно зростає з часом втрати об'єкта
        reinit_adaptive_rate: int - кількість кадрів для досягнення max threshold (default: 15)
            Після max_lost_frames threshold лінійно зростає від reinit_diou_threshold до reinit_diou_max
        verbose: bool - виводити деталі (default: False)

    Трифазна логіка:
        Фаза 1 (IoU Matching): Strict matching з last_valid_bbox
        Фаза 2 (DIoU Пошук + Очікування): Якщо IoU < threshold
            - Спробувати DIoU matching (якщо phase2_diou_threshold < 1.0)
            - Спробувати дострокову реініціалізацію (якщо waiting_reinit_conf_threshold < 1.0)
              - Якщо conf >= waiting_reinit_conf_threshold та diou >= waiting_reinit_diou_threshold
            - Інакше: режим очікування (grace period до max_lost_frames кадрів)
        Фаза 3 (Реідентифікація): Fallback до max(conf) detection з фільтрацією за DIoU
            - Спочатку перевірити високоякісну детекцію (якщо reinit_conf_threshold < 1.0)
            - Якщо reinit_diou_threshold > -1.0: вибирається max(conf) серед кандидатів з DIoU >= threshold
            - Якщо reinit_diou_threshold = -1.0: вибирається max(conf) без обмежень (default)

    VPE Quality Control:
        VPE збирається тільки якщо detection.conf >= vpe_conf_threshold
        Це забезпечує високу якість агрегованого VPE
    """

    def __init__(self,
                 model_path: str = 'yoloe-26s-seg.pt',
                 conf: float = 0.1,
                 conf_max: float = 0.1,
                 conf_adaptive_rate: int = 5,
                 imgsz: int = 640,
                 device: str = 'cuda',
                 iou_threshold: float = 0.3,
                 vpe_step: int = 10,
                 max_vpe: int = 5,
                 max_lost_frames: int = 30,
                 vpe_conf_threshold: float = 0.5,
                 vpe_conf_max: float = 0.5,
                 vpe_conf_adaptive_rate: int = 5,
                 warmup_frames: int = 10,
                 warmup_vpe_iou_threshold: float = 0.5,
                 phase2_diou_threshold: float = 1.0,
                 waiting_reinit_conf_threshold: float = 1.0,
                 waiting_reinit_diou_threshold: float = -0.5,
                 reinit_diou_threshold: float = -1.0,
                 reinit_diou_max: float = -0.9,
                 reinit_adaptive_rate: int = 15,
                 reinit_conf_threshold: float = 1.0,
                 phase3_redetection_validation_frames: int = 3,
                 phase3_appearance_weight: float = 0.0,
                 phase3_appearance_ref: str = 'aggregated',
                 vpe_gate_threshold: float = 0.0,
                 vpe_gate_ref: str = 'aggregated',
                 use_kalman: bool = False,
                 kalman_process_noise: float = 0.01,
                 kalman_measurement_noise: float = 10.0,
                 use_samurai_kalman: bool = False,
                 kalman_alpha_kf: float = 0.15,
                 kalman_tau_kf: int = 3,
                 kalman_n_max: int = 5,
                 kalman_iou_threshold: float = 0.3,
                 hybrid_conf_weight: float = 0.8,
                 use_dual_memory_vpe: bool = False,
                 dual_long_term_capacity: int = 5,
                 dual_long_term_quality_threshold: float = 0.7,
                 dual_long_term_weight: float = 0.4,
                 dual_short_term_capacity: int = 10,
                 dual_short_term_weight: float = 0.6,
                 dual_temporal_decay: float = 0.9,
                 dual_update_long_term_every: int = 50,
                 dual_replace_worst_lt: bool = True,
                 dual_use_anchor: bool = True,
                 dual_anchor_weight: float = 0.1,
                 verbose: bool = False,
                 debug_frame_step: int = 1,
                 **kwargs):
        super().__init__(**kwargs)

        if not YOLOE_AVAILABLE:
            raise ImportError("YOLOE not available. Please install ultralytics with YOLOE support.")

        # Перевірка сумісності моделі
        if '-pf' in model_path.lower():
            raise ValueError(
                f"❌ Модель '{model_path}' має суфікс '-pf' (prompt-free) та НЕ підтримує візуальні промти!\n"
                f"   Використовуйте моделі БЕЗ '-pf': yoloe-26s-seg.pt, yoloe-v8l-seg.pt, etc."
            )

        self.model_path = model_path
        self.conf = conf
        self.conf_max = conf_max
        self.conf_adaptive_rate = conf_adaptive_rate
        self.imgsz = imgsz
        self.device = device
        self.iou_threshold = iou_threshold
        self.vpe_step = vpe_step
        self.max_vpe = max_vpe
        self.max_lost_frames = max_lost_frames
        self.vpe_conf_threshold = vpe_conf_threshold
        self.vpe_conf_max = vpe_conf_max
        self.vpe_conf_adaptive_rate = vpe_conf_adaptive_rate
        self.warmup_frames = warmup_frames
        self.warmup_vpe_iou_threshold = warmup_vpe_iou_threshold
        self.phase2_diou_threshold = phase2_diou_threshold
        self.phase2_switch_validation_frames = kwargs.get('phase2_switch_validation_frames', 0)
        self.waiting_reinit_conf_threshold = waiting_reinit_conf_threshold
        self.waiting_reinit_diou_threshold = waiting_reinit_diou_threshold
        self.reinit_diou_threshold = reinit_diou_threshold
        self.reinit_diou_max = reinit_diou_max
        self.reinit_adaptive_rate = reinit_adaptive_rate
        self.reinit_conf_threshold = reinit_conf_threshold
        self.phase3_redetection_validation_frames = phase3_redetection_validation_frames
        # Phase 3 appearance-вибір кандидата: score = (1-w)*conf + w*sim(VPE)
        # w=0 -> вимкнено (max(conf), поведінка без змін); ref: 'aggregated' | 'anchor'
        self.phase3_appearance_weight = phase3_appearance_weight
        self.phase3_appearance_ref = phase3_appearance_ref
        # Гейт пам'яті: новий VPE додається лише якщо sim із референсом пам'яті
        # >= порога (шкала [0,1], (cos+1)/2); 0.0 = вимкнено. ref: 'aggregated'
        # (еволюціонує з виглядом; надійніше за 'anchor' при значних змінах)
        self.vpe_gate_threshold = vpe_gate_threshold
        self.vpe_gate_ref = vpe_gate_ref
        self.vpe_gate_accepted = 0   # діагностика
        self.vpe_gate_rejected = 0
        self.phase3_vpe_freeze_frames = kwargs.get('phase3_vpe_freeze_frames', 0)
        # Режим вибору reference bbox для Phase 3 порівнянь:
        #   'last_valid'    — остання детектована позиція (default, стабільно)
        #   'kalman_phase2' — остання Kalman предикція з Phase 2 (враховує рух при виході з Phase 2)
        #   'kalman_phase3' — поточна Kalman предикція у Phase 3 (найсвіжіша, але може дрейфувати)
        self.phase3_ref_mode = kwargs.get('phase3_ref_mode', 'last_valid')

        # ⭐ Phase 3 joint DIoU×conf score (за прапорцем use_joint_score):
        # один неперервний бал у [0,1] замість окремих conf/DIoU reinit-гейтів
        # + вибору max(conf). Об'єднує близькість і впевненість в одну ручку.
        #   score = geometric: conf^(1-λ) · diou_norm^λ  (AND-подібний, default)
        #           arithmetic: (1-λ)·conf + λ·diou_norm (з компенсацією)
        #   diou_norm = (diou+1)/2;  λ спадає від lam_start до lam_min з часом втрати.
        self.use_joint_score = kwargs.get('use_joint_score', False)
        self.joint_mode = kwargs.get('joint_mode', 'geometric')          # 'geometric' | 'arithmetic'
        self.reinit_joint_threshold = kwargs.get('reinit_joint_threshold', 0.5)
        self.joint_lam_start = kwargs.get('joint_lam_start', 0.5)         # вага близькості щойно після втрати
        self.joint_lam_min = kwargs.get('joint_lam_min', 0.2)            # вага близькості після довгої втрати
        self.reinit_diou_floor = kwargs.get('reinit_diou_floor', -0.95)  # жорсткий anti-teleport floor
        # Phase 2 (doubt/re-association після провалу IoU-матчу, до повної втрати):
        # зберігаємо просторовий gate diou>=phase2_diou_threshold (анти-дистрактор),
        # але серед плюсклих кандидатів обираємо за joint-балом, а не чистим max(DIoU).
        self.joint_apply_phase2 = kwargs.get('joint_apply_phase2', False)

        # Phase 1 High Confidence Re-ID
        self.phase1_high_conf_reid_threshold = kwargs.get('phase1_high_conf_reid_threshold', 0.9)
        self.phase1_high_conf_reid_iou = kwargs.get('phase1_high_conf_reid_iou', 0.3)
        self.phase1_high_conf_reid_diou = kwargs.get('phase1_high_conf_reid_diou', None)  # якщо задано — використовує DIoU замість IoU
        self.phase1_high_conf_reid_require_lost = kwargs.get('phase1_high_conf_reid_require_lost', False)
        self.phase1_high_conf_reid_validation_frames = kwargs.get('phase1_high_conf_reid_validation_frames', 0)
        self.phase1_high_conf_reid_validation_skip = kwargs.get('phase1_high_conf_reid_validation_skip', 0)
        self.phase1_high_conf_reid_conf_gap = kwargs.get('phase1_high_conf_reid_conf_gap', 0.0)
        self.use_kalman = use_kalman
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.use_samurai_kalman = use_samurai_kalman
        self.kalman_alpha_kf = kalman_alpha_kf
        self.kalman_tau_kf = kalman_tau_kf
        self.kalman_n_max = kalman_n_max
        self.kalman_iou_threshold = kalman_iou_threshold
        self.hybrid_conf_weight = hybrid_conf_weight
        self.use_dual_memory_vpe = use_dual_memory_vpe
        self.dual_long_term_capacity = dual_long_term_capacity
        self.dual_long_term_quality_threshold = dual_long_term_quality_threshold
        self.dual_long_term_weight = dual_long_term_weight
        self.dual_short_term_capacity = dual_short_term_capacity
        self.dual_short_term_weight = dual_short_term_weight
        self.dual_temporal_decay = dual_temporal_decay
        self.dual_update_long_term_every = dual_update_long_term_every
        self.dual_replace_worst_lt = dual_replace_worst_lt
        self.dual_use_anchor = dual_use_anchor
        self.dual_anchor_weight = dual_anchor_weight
        self.verbose = verbose
        self.debug_frame_step = debug_frame_step

        # Kalman filter (опціональний)
        self.kalman = None

        # ✨ SAMURAI Integration: режим вибору
        if self.use_samurai_kalman and not self.use_kalman:
            if self.verbose:
                print("⚠️  use_samurai_kalman встановлено, але use_kalman=False. Активую use_kalman...")
            self.use_kalman = True

        if self.use_samurai_kalman and self.verbose:
            print(f"✨ SAMURAI Kalman режим активований:")
            print(f"   α_kf={self.kalman_alpha_kf} ✓ (motion вага в Eq.7: IoU={self.kalman_alpha_kf:.1%}, affinity={(1-self.kalman_alpha_kf):.1%})")
            print(f"   τ_kf={self.kalman_tau_kf} ✓ (stability gate поріг для motion confidence)")
            print(f"   N_max={self.kalman_n_max} (memory bank розмір - на даний момент не реалізовано)")
            print(f"   hybrid_conf_weight={self.hybrid_conf_weight} (вага conf при збиранні VPE)")

        # Ініціалізація моделі
        if self.verbose:
            print(f"📦 Завантаження YOLOe моделі: {model_path}")
        self.model = YOLOE(model_path)
        if hasattr(self.model, 'to'):
            self.model.to(device)

        # VPE collection - dual memory or simple deque
        if self.use_dual_memory_vpe:
            self.dual_memory = DualMemoryVPE(
                long_term_capacity=dual_long_term_capacity,
                long_term_quality_threshold=dual_long_term_quality_threshold,
                long_term_weight=dual_long_term_weight,
                short_term_capacity=dual_short_term_capacity,
                short_term_weight=dual_short_term_weight,
                temporal_decay=dual_temporal_decay,
                update_long_term_every=dual_update_long_term_every,
                replace_worst_lt=dual_replace_worst_lt,
                use_anchor=dual_use_anchor,
                anchor_weight=dual_anchor_weight,
                verbose=verbose
            )
            self.vpe_list = None  # Not used in dual memory mode
            if self.verbose:
                print(f"🧠 Dual Memory VPE активовано:")
                print(f"   LT: capacity={dual_long_term_capacity}, quality>={dual_long_term_quality_threshold}, weight={dual_long_term_weight}")
                print(f"   ST: capacity={dual_short_term_capacity}, weight={dual_short_term_weight}, decay={dual_temporal_decay}")
                print(f"   Anchor: enabled={dual_use_anchor}, weight={dual_anchor_weight}")
                print(f"   LT update: every {dual_update_long_term_every} frames, replace_worst={dual_replace_worst_lt}")
        else:
            self.vpe_list = deque(maxlen=max_vpe)
            self.dual_memory = None

        self.frame_count = -1
        self.current_bbox = None
        self.aggregated_vpe = None
        self.initialized = False

        # Кешування set_classes: переустановлюємо класи лише коли агрегований VPE
        # реально оновився (_vpe_version змінюється в _aggregate_vpe)
        self._vpe_version = 0
        self._applied_vpe_version = None
        # Виділений предиктор для видобування VPE (fix: 1 forward замість 2)
        self._vpe_predictor = None

        # Трифазна логіка tracking
        self.last_valid_bbox = None  # Останній валідний bbox (для IoU порівняння)
        self.lost_frames = 0  # Лічильник кадрів без успішного matching

        # VPE quality control
        self.vpe_pending = False  # Флаг: чи потрібно зібрати VPE при достатній conf

        # Warmup period для агресивного збору VPE
        self.in_warmup = False  # Чи знаходимось у warmup періоді
        self.warmup_vpe_collected = 0  # Кількість VPE зібраних під час warmup

        # Debug info для візуалізації
        self.rejected_candidates = []  # Відкинуті кандидати у Фазі 3
        self.search_candidates = []  # Всі detections під час Phase 2/3 для візуалізації
        self.top_candidates = []  # Top candidates з IoU scores у Phase 1 для візуалізації
        self.last_bbox_is_kalman_only = False  # Чи є поточний bbox тільки від Калмана (не валідовано детекціями)
        self.current_frame_detections = []  # Всі детекції з proximity інформацією на поточному кадрі
        self.selected_detection_idx = None  # Індекс вибраної детекції в Phase 1

        # Phase 3 Redetection Validation Gate
        self.in_phase3_validation = False  # Чи ми в режимі validation після Phase 3 реініціалізації
        self.phase3_validation_consecutive_successes = 0  # Лічильник послідовних успішних фреймів у validation режимі
        self.phase3_validation_failure_count = 0  # Лічильник невдалих кадрів під час validation (незалежний від lost_frames)
        self.phase3_validation_bbox = None  # Bbox що перевіряється у validation режимі
        self.phase3_reference_bbox = None  # Bbox з яким порівнюються кандидати (для візуалізації)
        self.phase3_kalman_exit_bbox = None  # Остання Kalman предикція з Phase 2 (xyxy)

        # Phase 3 VPE Freeze: блокувати збір VPE після Phase 3 reinit
        self.phase3_vpe_freeze_counter = 0  # Кількість кадрів що залишилось блокувати VPE

        # Phase 1 High Confidence Re-ID Validation
        self.phase1_high_conf_reid_candidate = None  # Кандидат для high-conf re-ID (dict з bbox, conf, iou)
        self.phase1_high_conf_reid_validation_count = 0  # Лічильник успішних перевірок
        self.phase1_high_conf_reid_skip_counter = 0    # Кадрів залишилось пропустити (0 = кадр перевірки)

        # Phase 2 Switch Validation
        self.phase2_pending_candidate = None  # Pending кандидат для перемикання в Phase 2 (dict з bbox, conf, diou)
        self.phase2_pending_validation_count = 0  # Лічильник кадрів очікування перед перемиканням

        if self.verbose:
            # Conf adaptive info
            if conf_max > conf:
                conf_msg = f", adaptive_conf={conf}→{conf_max} ({conf_adaptive_rate}vpe)"
            else:
                conf_msg = f", conf>={conf}"

            # VPE conf adaptive info
            if vpe_conf_max > vpe_conf_threshold:
                vpe_conf_msg = f", adaptive_vpe_conf={vpe_conf_threshold}→{vpe_conf_max} ({vpe_conf_adaptive_rate}vpe)"
            else:
                vpe_conf_msg = f", vpe_conf>={vpe_conf_threshold}"

            # Phase 2 DIoU info
            if phase2_diou_threshold < 1.0:
                phase2_msg = f", phase2_diou>={phase2_diou_threshold}"
            else:
                phase2_msg = ""

            # Waiting reinit info (Phase 2 early reinit)
            if waiting_reinit_conf_threshold < 1.0:
                waiting_reinit_msg = f", waiting_reinit(conf>={waiting_reinit_conf_threshold},diou>={waiting_reinit_diou_threshold})"
            else:
                waiting_reinit_msg = ""

            # DIoU adaptive info
            if reinit_diou_threshold > -1.0:
                if reinit_diou_max < reinit_diou_threshold:
                    # reinit_diou_max менше (більш негативне) = менш строге
                    reinit_msg = f", adaptive_diou={reinit_diou_threshold}→{reinit_diou_max} ({reinit_adaptive_rate}fr)"
                else:
                    reinit_msg = f", reinit_diou>={reinit_diou_threshold}"
            else:
                reinit_msg = ""

            # Reinit conf threshold info
            if reinit_conf_threshold < 1.0:
                reinit_conf_msg = f", reinit_conf>={reinit_conf_threshold}"
            else:
                reinit_conf_msg = ""

            # Phase 3 Validation info
            phase3_validation_msg = f", phase3_validation={phase3_redetection_validation_frames}fr"

            # Warmup info
            warmup_msg = f", warmup={warmup_frames}fr (IoU>={warmup_vpe_iou_threshold})"

            # Kalman info
            if use_kalman:
                kalman_msg = f", kalman=ON (process={kalman_process_noise}, measure={kalman_measurement_noise})"
            else:
                kalman_msg = ""

            print(f"✅ YOLOe-VP-IoU готовий (vpe_step={vpe_step}, max_vpe={max_vpe}, iou_threshold={iou_threshold}, max_lost_frames={max_lost_frames}{conf_msg}{vpe_conf_msg}{warmup_msg}{phase2_msg}{waiting_reinit_msg}{reinit_msg}{reinit_conf_msg}{phase3_validation_msg}{kalman_msg})")

    @classmethod
    def get_name(cls) -> str:
        return "YOLOe-VP-IoU"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'model_path': 'yoloe-26s-seg.pt',
            'conf': 0.1,
            'imgsz': 640,
            'device': 'cuda',
            'iou_threshold': 0.3,
            'vpe_step': 10,
            'max_vpe': 5,
            'max_lost_frames': 30,
            'vpe_conf_threshold': 0.5,
            'warmup_frames': 10,
            'warmup_vpe_iou_threshold': 0.5,
            'phase2_diou_threshold': 1.0,
            'waiting_reinit_conf_threshold': 1.0,
            'waiting_reinit_diou_threshold': -0.5,
            'reinit_diou_threshold': -1.0,
            'reinit_diou_max': -0.9,
            'reinit_adaptive_rate': 15,
            'reinit_conf_threshold': 1.0,
            'phase3_redetection_validation_frames': 3,
            'phase3_appearance_weight': 0.0,
            'phase3_appearance_ref': 'aggregated',
            'vpe_gate_threshold': 0.0,
            'vpe_gate_ref': 'aggregated',
            'use_kalman': False,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 10.0,
            'verbose': False,
            'debug_frame_step': 1,
        }

    def initialize(self, image: np.ndarray, bbox: List[float]) -> bool:
        """
        Ініціалізація з першим кадром

        Args:
            image: Перший кадр (H, W, 3) BGR
            bbox: Початковий bbox [x, y, w, h]

        Returns:
            True якщо успішно
        """
        try:
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            self.current_bbox = [x1, y1, x2, y2]
            self.last_valid_bbox = [x1, y1, x2, y2]  # Перший bbox - валідний
            self.lost_frames = 0  # Reset counter

            if self.verbose:
                print(f"✅ Ініціалізація з bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

            # Зібрати перший VPE
            if self.verbose:
                print(f"🔄 Кадр 1: Збір VPE [INIT] (VPE=1/{self.max_vpe})")
            self._collect_vpe(image, self.current_bbox, conf=1.0)  # Use conf=1.0 for initialization

            # Ініціалізація Калман фільтру якщо увімкнено
            if self.use_kalman:
                self.kalman = KalmanBoxTracker(
                    bbox=[x, y, w, h],
                    process_noise=self.kalman_process_noise,
                    measurement_noise=self.kalman_measurement_noise
                )
                if self.verbose:
                    print(f"   🎯 Калман фільтр ініціалізовано (process_noise={self.kalman_process_noise}, measurement_noise={self.kalman_measurement_noise})")

            self.initialized = True
            self.frame_count = 1

            # Початок warmup періоду
            self.in_warmup = True
            self.warmup_vpe_collected = 0

            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Помилка ініціалізації: {e}")
            return False

    def _update_kalman_with_detection(self, detected_bbox_xywh: List[float], box_conf: float,
                                      all_detections: Optional[List] = None) -> Tuple[List[float], Optional[dict]]:
        """
        Оновити Калман фільтр з детекцією з підтримкою обох режимів (старий і SAMURAI)

        ✨ Phase 3: YOLOeVPIoUTracker Integration

        Args:
            detected_bbox_xywh: [x, y, w, h] - детектований бокс
            box_conf: float - впевненість детектора
            all_detections: Optional[List] - всі детекції для SAMURAI гібридної оцінки

        Returns:
            Tuple[smoothed_bbox, samurai_info]:
                smoothed_bbox: [x, y, w, h] або xyxy
                samurai_info: dict з SAMURAI статистикою (або None для старого режиму)
        """
        if not self.use_kalman or self.kalman is None:
            return detected_bbox_xywh, None

        # ✨ SAMURAI Режим: IoU scoring + Stability gate + Hybrid scoring
        if self.use_samurai_kalman and all_detections is not None and len(all_detections) > 0:
            try:
                # Phase 1: Обчислити IoU scores для всіх детекцій (Equation 6)
                kalman_pred, iou_scores = self.kalman.predict_with_iou_scores(
                    [d['bbox_xywh'] for d in all_detections]
                )

                # Phase 2: Симуляція affinity scores (реально від SAM2)
                # На даний момент використовуємо conf як наближення affinity
                affinity_scores = np.array([d['conf'] for d in all_detections])

                # Phase 2: Обчислити гібридну оцінку (Equation 7 з SAMURAI)
                # M* = argmax_i(α_kf · s_kf(M_i) + (1-α_kf) · s_mask(M_i))
                # α_kf = 0.15 означає: IoU отримує 15%, affinity отримує 85%
                # (на відміну від hybrid_conf_weight який контролює VPE conf при збиранні)
                hybrid_scores = (
                    self.kalman_alpha_kf * iou_scores +
                    (1 - self.kalman_alpha_kf) * affinity_scores
                )

                # hybrid_scores = (
                #         (
                #                     1 - self.hybrid_conf_weight) * iou_scores +  # ❌ Неправильний параметр
                #         self.hybrid_conf_weight * affinity_scores
                # )

                # Визначити чи було успішне оновлення
                best_idx = np.argmax(hybrid_scores)
                is_successful = (iou_scores[best_idx] >= self.kalman_iou_threshold)

                if self.verbose:
                    # Логування всіх SAMURAI кандидатів
                    sorted_indices = np.argsort(hybrid_scores)[::-1]
                    for rank, idx in enumerate(sorted_indices):
                        conf = all_detections[idx]['conf']
                        status = "BEST" if idx == best_idx else ""
                        print(f"   ✨ SAMURAI Box {rank+1} (idx={idx}): IoU={iou_scores[idx]:.3f}, "
                              f"affinity={affinity_scores[idx]:.3f}, hybrid={hybrid_scores[idx]:.3f}, "
                              f"conf={conf:.3f} {status}")

                    motion_conf = compute_motion_confidence(
                        self.kalman.successful_frames,
                        tau_kf=self.kalman_tau_kf
                    )
                    print(f"   ✨ SAMURAI SELECTED: Box {best_idx} | IoU={iou_scores[best_idx]:.3f}, "
                          f"affinity={affinity_scores[best_idx]:.3f}, hybrid={hybrid_scores[best_idx]:.3f}, "
                          f"motion_conf={motion_conf:.2f}")

                # Phase 1: Stability gate оновлення
                smoothed_bbox = self.kalman.update(detected_bbox_xywh, is_successful=is_successful)

                samurai_info = {
                    'iou_scores': iou_scores,
                    'affinity_scores': affinity_scores,
                    'hybrid_scores': hybrid_scores,
                    'best_idx': best_idx,
                    'is_successful': is_successful,
                    'motion_active': self.kalman.use_motion,
                    'successful_frames': self.kalman.successful_frames
                }

                return smoothed_bbox, samurai_info

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  SAMURAI режим помилка: {e}, fallback на стандартний режим")
                # Fallback на стандартний режим
                smoothed_bbox = self.kalman.update(detected_bbox_xywh, is_successful=True)
                return smoothed_bbox, None

        # Стандартний (старий) режим: просто оновити Калман
        else:
            smoothed_bbox = self.kalman.update(detected_bbox_xywh, is_successful=True)
            return smoothed_bbox, None

    def update(self, image: np.ndarray) -> Tuple[bool, Optional[List[float]]]:
        """
        Оновлення на новому кадрі з трифазною логікою

        Фаза 1: IoU Matching - порівняння з last_valid_bbox
        Фаза 2: DIoU Пошук + Дострокова Реініціалізація + Очікування
            - DIoU matching для швидкорухомих об'єктів
            - Early reinit якщо conf >= waiting_reinit_conf_threshold та diou >= waiting_reinit_diou_threshold
            - Grace period до max_lost_frames кадрів
        Фаза 3: Реідентифікація - fallback до detection з max(conf)

        Args:
            image: Новий кадр (H, W, 3) BGR

        Returns:
            (success, bbox) - успіх та bbox [x, y, w, h]
        """
        if not self.initialized:
            return False, None

        # Ініціалізація box_conf для VPE collection
        box_conf = 0.0  # Default value, буде оновлено при знаходженні detection

        self.frame_count += 1

        # Очистити candidates з попереднього фрейму на початку кожного update
        self.top_candidates = []
        self.rejected_candidates = []
        self.search_candidates = []
        self.current_frame_detections = []

        # Зменшити VPE freeze лічильник якщо активний
        if self.phase3_vpe_freeze_counter > 0:
            self.phase3_vpe_freeze_counter -= 1
            if self.verbose and self.phase3_vpe_freeze_counter == 0:
                print(f"🔓 Кадр {self.frame_count}: Phase 3 VPE freeze знято, збір VPE відновлено")

        # Перевірка завершення warmup періоду
        if self.in_warmup and self.frame_count >= self.warmup_frames:
            self.in_warmup = False
            if self.verbose:
                print(f"🎯 Warmup завершено (зібрано {self._get_vpe_count()} VPE за {self.warmup_frames} кадрів)")

        # Калман прогноз (якщо увімкнено)
        kalman_prediction = None
        if self.use_kalman and self.kalman is not None:
            kalman_prediction = self.kalman.predict()
            if self.verbose:
                x, y, w, h = kalman_prediction
                print(f"🔮 Кадр {self.frame_count}: Калман прогноз [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")

        try:
            # ========================================
            # VP Detection (знаходження кандидатів)
            # ========================================
            use_vpe = False
            if self.aggregated_vpe is not None:
                # Validate shape before using [1, 1, D]
                if self.verbose:
                    print(f"   🔍 Aggregated VPE shape: {self.aggregated_vpe.shape}")

                if self.aggregated_vpe.dim() == 3 and self.aggregated_vpe.size(0) == 1 and self.aggregated_vpe.size(1) == 1:
                    use_vpe = True
                else:
                    if self.verbose:
                        print(f"   ⚠️  Invalid aggregated VPE shape: {self.aggregated_vpe.shape}, очікується [1, 1, D], falling back to visual prompts")

            if use_vpe:
                # Використати агрегований VPE; set_classes — лише коли агрегат
                # оновився після _aggregate_vpe (а не на кожному кадрі)
                if self._applied_vpe_version != self._vpe_version:
                    self.model.is_fused = lambda: False
                    self.model.set_classes([0], self.aggregated_vpe)  # Use int instead of string
                    self._applied_vpe_version = self._vpe_version

                current_conf = self._get_adaptive_conf()
                results = self.model.predict(
                    image,
                    conf=current_conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            else:
                # Fallback: звичайні visual prompts
                visual_prompts = dict(
                    bboxes=np.array([self.current_bbox]),
                    cls=np.array([0]),
                )

                current_conf = self._get_adaptive_conf()
                results = self.model.predict(
                    image,
                    visual_prompts=visual_prompts,
                    predictor=YOLOEVPSegPredictor,
                    conf=current_conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
                # VP-предикт переналаштовує nc/names моделі — наступного разу
                # за use_vpe класи треба встановити наново
                self._applied_vpe_version = None

            if len(results) == 0 or len(results[0].boxes) == 0:
                # Немає detections
                self.lost_frames += 1

                # ⭐ Скинути Phase 3 validation якщо немає detections взагалі
                # (потрібно строго N послідовних кадрів БЕЗ пропусків)
                if self.in_phase3_validation:
                    if self.verbose:
                        print(f"❌ Кадр {self.frame_count}: [PHASE 3 VAL] скасовано через відсутність detections (consecutive={self.phase3_validation_consecutive_successes})")
                    self.in_phase3_validation = False
                    self.phase3_validation_bbox = None
                    self.phase3_validation_consecutive_successes = 0
                    self.phase3_validation_failure_count = 0

                if self.verbose:
                    print(f"⚠️  Кадр {self.frame_count}: No detections (lost={self.lost_frames}/{self.max_lost_frames})")

                if self.lost_frames >= self.max_lost_frames:
                    # Занадто довго без detections - fail
                    if self.verbose:
                        print(f"❌ Кадр {self.frame_count}: Tracking lost (no detections)")
                    return False, None
                else:
                    # Повернути прогноз Калмана або last_valid_bbox (очікування)
                    if self.use_kalman and kalman_prediction is not None:
                        if self.verbose:
                            print(f"   🔮 Використовуємо Калман прогноз (no detections)")
                        # Оновити current_bbox на Kalman prediction для неперервності траєкторії
                        x, y, w, h = kalman_prediction  # xywh
                        self.current_bbox = [x, y, x + w, y + h]  # convert to xyxy
                        self.last_bbox_is_kalman_only = True  # Відмітити, що це не валідовано детекціями
                        return True, kalman_prediction
                    else:
                        x1, y1, x2, y2 = self.last_valid_bbox
                        self.current_bbox = [x1, y1, x2, y2]  # Also update for consistency
                        self.last_bbox_is_kalman_only = False
                        return True, [x1, y1, x2 - x1, y2 - y1]

            boxes = results[0].boxes

            # Ініціалізувати - розрахуємо proximity після Phase 1
            self.current_frame_detections = []
            self.selected_detection_idx = None

            # ========================================
            # Фаза 1: IoU Matching з last_valid_bbox
            # ========================================
            best_iou_idx = -1
            best_iou = 0
            best_conf_idx = -1
            best_conf = 0

            # Зберегти інформацію про всі кандидати для логування
            all_candidates_info = []

            for idx, box in enumerate(boxes):
                box_xyxy = box.xyxy[0].cpu().numpy()
                box_conf = float(box.conf[0].cpu().numpy())

                # Обчислити IoU з last_valid_bbox (не з current!)
                iou = self._compute_iou(self.last_valid_bbox, box_xyxy)

                # Перевірка розміру: відхилити детекції зі значною зміною розміру
                last_valid_w = self.last_valid_bbox[2] - self.last_valid_bbox[0]
                last_valid_h = self.last_valid_bbox[3] - self.last_valid_bbox[1]
                box_w = box_xyxy[2] - box_xyxy[0]
                box_h = box_xyxy[3] - box_xyxy[1]

                size_ratio_w = box_w / (last_valid_w + 1e-6)
                size_ratio_h = box_h / (last_valid_h + 1e-6)

                # Допускаємо зміну розміру в діапазоні 0.3-2.5 (30%-250%)
                size_valid = (0.3 <= size_ratio_w <= 2.5) and (0.3 <= size_ratio_h <= 2.5)

                # Зберегти інформацію про кандидата
                all_candidates_info.append({
                    'idx': idx,
                    'conf': box_conf,
                    'iou': iou,
                    'size_ratio_w': size_ratio_w,
                    'size_ratio_h': size_ratio_h,
                    'size_valid': size_valid,
                    'box_xyxy': box_xyxy
                })

                if iou > best_iou and size_valid:
                    best_iou = iou
                    best_iou_idx = idx

                if box_conf > best_conf:
                    best_conf = box_conf
                    best_conf_idx = idx

            # Логування всіх кандидатів для звичайного IoU
            if self.verbose and not self.use_samurai_kalman and len(all_candidates_info) > 0:
                for cand in all_candidates_info:
                    if cand['size_valid']:
                        print(f"   ✅ Box {cand['idx']}: IoU={cand['iou']:.3f}, conf={cand['conf']:.3f} (W/H={cand['size_ratio_w']:.2f}/{cand['size_ratio_h']:.2f})")
                    else:
                        print(f"   ⚠️  Box {cand['idx']}: IoU={cand['iou']:.3f}, conf={cand['conf']:.3f} (W/H={cand['size_ratio_w']:.2f}/{cand['size_ratio_h']:.2f}) - size_ratio невалідна")

            # ========================================
            # Phase 1 High Confidence Re-ID (Pre-check)
            # ========================================
            # Перевірити чи є детекція з екстремально високою впевненістю
            # ЦЕ ВИКОНУЄТЬСЯ ПЕРЕД перевіркою IoU threshold!
            high_conf_override = False

            # Перевірка умови lost_frames (опціонально)
            should_check_high_conf = (
                self.phase1_high_conf_reid_threshold < 1.0 and
                (not self.phase1_high_conf_reid_require_lost or self.lost_frames > 0)
            )

            if should_check_high_conf:
                # Отримати впевненість поточного best match (якщо є)
                current_best_conf = 0.0
                if best_iou_idx >= 0 and best_iou_idx < len(boxes):
                    current_best_conf = float(boxes[best_iou_idx].conf[0].cpu().numpy())

                high_conf_candidates = []
                for idx, box in enumerate(boxes):
                    box_xyxy_temp = box.xyxy[0].cpu().numpy()
                    box_conf_temp = float(box.conf[0].cpu().numpy())

                    # Перевірити високу впевненість ТА порівняти з поточним (з gap)
                    conf_gap_threshold = current_best_conf + self.phase1_high_conf_reid_conf_gap
                    if (box_conf_temp >= self.phase1_high_conf_reid_threshold and
                        box_conf_temp > conf_gap_threshold):
                        # Обчислити DIoU або IoU з last_valid_bbox
                        if self.phase1_high_conf_reid_diou is not None:
                            proximity = self._compute_diou(self.last_valid_bbox, box_xyxy_temp)
                            passes_proximity = proximity >= self.phase1_high_conf_reid_diou
                        else:
                            proximity = self._compute_iou(self.last_valid_bbox, box_xyxy_temp)
                            passes_proximity = proximity >= self.phase1_high_conf_reid_iou

                        if passes_proximity:
                            high_conf_candidates.append({
                                'idx': idx,
                                'bbox': box_xyxy_temp,
                                'conf': box_conf_temp,
                                'iou': proximity
                            })

                # Якщо знайдено high-conf кандидата
                if len(high_conf_candidates) > 0:
                    # Взяти кандидата з найвищою впевненістю
                    best_high_conf = max(high_conf_candidates, key=lambda x: x['conf'])

                    # Перевірка чи потрібна temporal validation
                    if self.phase1_high_conf_reid_validation_frames > 0:
                        # ===== TEMPORAL VALIDATION MODE (з підтримкою skip) =====
                        skip = self.phase1_high_conf_reid_validation_skip

                        # Перевірити чи це той самий кандидат що і раніше
                        is_same_candidate = False
                        if self.phase1_high_conf_reid_candidate is not None:
                            prev_bbox = self.phase1_high_conf_reid_candidate['bbox']
                            curr_bbox = best_high_conf['bbox']
                            candidate_iou = self._compute_iou(prev_bbox, curr_bbox)
                            is_same_candidate = candidate_iou > 0.5

                        if is_same_candidate:
                            if self.phase1_high_conf_reid_skip_counter > 0:
                                # ===== SKIP FRAME: пропускаємо, тільки оновлюємо позицію =====
                                self.phase1_high_conf_reid_skip_counter -= 1
                                # Оновлюємо позицію кандидата (слідкуємо за рухом)
                                self.phase1_high_conf_reid_candidate = best_high_conf
                                if self.verbose:
                                    print(f"⏭️  Кадр {self.frame_count}: [PHASE 1] High-Conf Re-ID пропуск ({self.phase1_high_conf_reid_skip_counter} залишилось), count={self.phase1_high_conf_reid_validation_count}/{self.phase1_high_conf_reid_validation_frames}")
                            else:
                                # ===== CHECK FRAME: перевіряємо та рахуємо =====
                                self.phase1_high_conf_reid_validation_count += 1
                                # Оновити збережений кандидат
                                self.phase1_high_conf_reid_candidate = best_high_conf
                                if self.verbose:
                                    skip_info = f", skip={skip}" if skip > 0 else ""
                                    print(f"🔄 Кадр {self.frame_count}: [PHASE 1] High-Conf Re-ID перевірка {self.phase1_high_conf_reid_validation_count}/{self.phase1_high_conf_reid_validation_frames}{skip_info}")
                                    print(f"   Кандидат: conf={best_high_conf['conf']:.3f}, proximity={best_high_conf['iou']:.3f}")

                                if self.phase1_high_conf_reid_validation_count >= self.phase1_high_conf_reid_validation_frames:
                                    # Валідація успішна - перемкнутися!
                                    if self.verbose:
                                        print(f"✅ Кадр {self.frame_count}: [PHASE 1] High-Conf Re-ID Override! (валідація пройдена)")
                                        print(f"   Поточний: conf={current_best_conf:.3f}, best_iou={best_iou:.3f}")
                                        print(f"   Новий: conf={best_high_conf['conf']:.3f}, proximity={best_high_conf['iou']:.3f}")
                                        print(f"   Переключення на high-conf детекцію (Δconf={best_high_conf['conf']-current_best_conf:.3f})")

                                    best_iou_idx = best_high_conf['idx']
                                    best_iou = best_high_conf['iou']
                                    high_conf_override = True

                                    # Скинути валідацію
                                    self.phase1_high_conf_reid_candidate = None
                                    self.phase1_high_conf_reid_validation_count = 0
                                    self.phase1_high_conf_reid_skip_counter = 0
                                else:
                                    # Перевірка пройдена, але ще не досягнуто порогу
                                    # Запустити skip-період перед наступною перевіркою
                                    self.phase1_high_conf_reid_skip_counter = skip
                        else:
                            # Новий кандидат - почати валідацію спочатку
                            self.phase1_high_conf_reid_candidate = best_high_conf
                            self.phase1_high_conf_reid_validation_count = 1
                            self.phase1_high_conf_reid_skip_counter = skip  # Перший skip-період
                            if self.verbose:
                                skip_info = f", skip={skip}" if skip > 0 else ""
                                print(f"🆕 Кадр {self.frame_count}: [PHASE 1] High-Conf Re-ID новий кандидат: 1/{self.phase1_high_conf_reid_validation_frames}{skip_info}")
                                print(f"   conf={best_high_conf['conf']:.3f}, proximity={best_high_conf['iou']:.3f}")
                    else:
                        # ===== IMMEDIATE MODE (без validation) =====
                        if self.verbose:
                            print(f"🔄 Кадр {self.frame_count}: [PHASE 1] High-Conf Re-ID Override!")
                            print(f"   Поточний: conf={current_best_conf:.3f}, best_iou={best_iou:.3f}")
                            print(f"   Новий: conf={best_high_conf['conf']:.3f}, proximity={best_high_conf['iou']:.3f}")
                            print(f"   Переключення на high-conf детекцію (Δconf={best_high_conf['conf']-current_best_conf:.3f})")

                        best_iou_idx = best_high_conf['idx']
                        best_iou = best_high_conf['iou']
                        high_conf_override = True
                else:
                    # Кандидатів не знайдено на цьому кадрі
                    if self.phase1_high_conf_reid_candidate is not None:
                        if self.phase1_high_conf_reid_skip_counter > 0:
                            # ===== SKIP FRAME без детекції: пропускаємо (не скидаємо) =====
                            self.phase1_high_conf_reid_skip_counter -= 1
                            if self.verbose:
                                print(f"⏭️  Кадр {self.frame_count}: [PHASE 1] High-Conf Re-ID пропуск без детекції ({self.phase1_high_conf_reid_skip_counter} залишилось)")
                        else:
                            # CHECK FRAME без детекції: скидаємо валідацію
                            if self.verbose:
                                print(f"❌ Кадр {self.frame_count}: [PHASE 1] High-Conf Re-ID скинуто (кандидат не знайдено на кадрі перевірки)")
                            self.phase1_high_conf_reid_candidate = None
                            self.phase1_high_conf_reid_validation_count = 0
                            self.phase1_high_conf_reid_skip_counter = 0

            # Перевірка IoU threshold та розміру (або high_conf_override)
            if best_iou >= self.iou_threshold or high_conf_override:
                # ========================================
                # ФАЗА 1: УСПІШНИЙ IoU MATCHING
                # ========================================
                best_box = boxes[best_iou_idx]
                box_xyxy = best_box.xyxy[0].cpu().numpy()
                box_conf = float(best_box.conf[0].cpu().numpy())

                # Оновити bbox (з Калманом якщо увімкнено)
                detected_bbox_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]
                samurai_info = None  # Ініціалізувати
                if self.use_kalman and self.kalman is not None:
                    # ✨ Phase 3: Зібрати всі детекції для SAMURAI гібридної оцінки
                    all_detections = []
                    if self.use_samurai_kalman:
                        for idx, box in enumerate(boxes):
                            box_xyxy_temp = box.xyxy[0].cpu().numpy()
                            box_conf_temp = float(box.conf[0].cpu().numpy())
                            all_detections.append({
                                'idx': idx,
                                'bbox_xywh': [
                                    float(box_xyxy_temp[0]),
                                    float(box_xyxy_temp[1]),
                                    float(box_xyxy_temp[2] - box_xyxy_temp[0]),
                                    float(box_xyxy_temp[3] - box_xyxy_temp[1])
                                ],
                                'conf': box_conf_temp
                            })

                    # Оновити Калман з детекцією та отримати згладжений результат
                    smoothed_bbox, samurai_info = self._update_kalman_with_detection(
                        detected_bbox_xywh, box_conf, all_detections if self.use_samurai_kalman else None
                    )
                    x, y, w, h = smoothed_bbox
                    self.current_bbox = [x, y, x + w, y + h]  # Конвертувати назад у xyxy
                else:
                    self.current_bbox = box_xyxy.tolist()

                self.last_valid_bbox = self.current_bbox  # Оновити валідний bbox
                self.lost_frames = 0  # Reset counter
                self.last_bbox_is_kalman_only = False  # Детекція знайдена - bbox валідовано

                # ⭐ Скинути Phase 3 validation якщо вона була активна
                # (Phase 1 IoU commit перериває Phase 3 validation, тому скидаємо лічильник)
                if self.in_phase3_validation:
                    if self.verbose:
                        print(f"   🔄 [PHASE 1] Phase 3 validation скинута (Phase 1 commit перериває validation)")
                    self.in_phase3_validation = False
                    self.phase3_validation_bbox = None
                    self.phase3_validation_consecutive_successes = 0
                    self.phase3_validation_failure_count = 0

                # Скасувати Phase 2 pending якщо справжній об'єкт з'явився
                if self.phase2_pending_candidate is not None:
                    if self.verbose:
                        print(f"   ✅ [PHASE 2] Pending скасовано - справжній об'єкт знайдено (IoU={best_iou:.3f})")
                    self.phase2_pending_candidate = None
                    self.phase2_pending_validation_count = 0

                # Зберегти індекс вибраної детекції для proximity таблиці
                self.selected_detection_idx = best_iou_idx

                # Розрахувати proximity для всіх детекцій (тільки коли таблиця буде показана)
                if self.frame_count % self.debug_frame_step == 0:
                    self.current_frame_detections = self._get_all_detections_proximity(image, boxes, best_iou_idx)
                    if self.verbose and len(self.current_frame_detections) > 0:
                        print(f"   📊 Зібрано proximity для {len(self.current_frame_detections)} детекцій")

                # (Phase 3 validation відбувається всередині Phase 3, не тут)

                # Зберегти candidates для візуалізації
                self.top_candidates = []

                if self.use_samurai_kalman and samurai_info and 'iou_scores' in samurai_info:
                    # Для SAMURAI: зберігати ВСІ кандидати з SAMURAI IoU scores, сортовані по hybrid score
                    iou_scores = samurai_info['iou_scores']
                    affinity_scores = samurai_info.get('affinity_scores', np.zeros_like(iou_scores))
                    hybrid_scores = samurai_info.get('hybrid_scores', iou_scores)
                    best_idx_samurai = samurai_info.get('best_idx', np.argmax(hybrid_scores))

                    # Зберегти conf від best боксу SAMURAI для VPE collection
                    box_conf = 0.0  # Default
                    if best_idx_samurai < len(boxes):
                        best_box = boxes[best_idx_samurai]
                        box_conf = float(best_box.conf[0].cpu().numpy())

                    if len(iou_scores) > 0:
                        sorted_indices = np.argsort(hybrid_scores)[::-1]
                        for rank, idx in enumerate(sorted_indices):
                            if idx < len(boxes):
                                box = boxes[idx]
                                box_xyxy = box.xyxy[0].cpu().numpy()
                                curr_box_conf = float(box.conf[0].cpu().numpy())
                                x1, y1, x2, y2 = box_xyxy
                                is_best = (idx == best_idx_samurai)
                                self.top_candidates.append({
                                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                                    'iou': float(iou_scores[idx]),
                                    'affinity': float(affinity_scores[idx]),
                                    'hybrid': float(hybrid_scores[idx]),
                                    'conf': curr_box_conf,
                                    'is_best_match': is_best,
                                    'type': 'samurai'
                                })
                else:
                    # Для звичайного IoU: зберігати ВСІ кандидати з IoU до last_valid_bbox, сортовані по IoU
                    if len(all_candidates_info) > 0:
                        sorted_candidates = sorted(all_candidates_info, key=lambda x: x['iou'], reverse=True)
                        for cand in sorted_candidates:
                            x1, y1, x2, y2 = cand['box_xyxy']
                            self.top_candidates.append({
                                'bbox': [x1, y1, x2 - x1, y2 - y1],
                                'iou': cand['iou'],
                                'conf': cand['conf'],
                                'is_best_match': (cand['idx'] == best_iou_idx),
                                'size_valid': cand['size_valid'],
                                'type': 'standard'
                            })

                self.search_candidates = []  # Очистити search candidates (Phase 2/3)

                if self.verbose:
                    if self.use_kalman:
                        if self.use_samurai_kalman:
                            print(f"✅ Кадр {self.frame_count}: [PHASE 1] IoU Match + ✨SAMURAI-Kalman (IoU={best_iou:.3f}, Conf={box_conf:.3f})")
                        else:
                            print(f"✅ Кадр {self.frame_count}: [PHASE 1] IoU Match + Kalman (IoU={best_iou:.3f}, Conf={box_conf:.3f})")
                    else:
                        print(f"✅ Кадр {self.frame_count}: [PHASE 1] IoU Match (IoU={best_iou:.3f}, Conf={box_conf:.3f})")

                # Зібрати VPE якщо час (або pending) та conf достатня
                # Під час warmup збирати кожен кадр
                # Блокувати якщо Phase 3 VPE freeze активний
                should_collect_vpe = (self.phase3_vpe_freeze_counter == 0) and (
                    self.in_warmup or (self.frame_count % self.vpe_step == 0) or self.vpe_pending
                )

                if should_collect_vpe:
                    # Під час warmup використовувати мінімальний поріг (той що використовується для детекції)
                    if self.in_warmup:
                        current_vpe_threshold = self._get_adaptive_conf()  # Той самий conf що для детекції
                    else:
                        current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()

                    if box_conf >= current_vpe_threshold:
                        if self.verbose:
                            pending_msg = " (pending)" if self.vpe_pending else ""
                            warmup_msg = " [WARMUP]" if self.in_warmup else ""
                            print(f"🔄 Кадр {self.frame_count}: Збір VPE{warmup_msg}{pending_msg} (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={self._get_vpe_count()}/{self.max_vpe})")
                        self._collect_vpe(image, self.current_bbox, box_conf)
                        if self.in_warmup:
                            self.warmup_vpe_collected += 1
                        self.vpe_pending = False  # Зібрано, скинути флаг
                    else:
                        self.vpe_pending = True  # Встановити флаг для наступних кадрів
                        if self.verbose:
                            print(f"⚠️  Кадр {self.frame_count}: VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={self._get_vpe_count()}/{self.max_vpe}), спроба на наступному кадрі")

                x1, y1, x2, y2 = self.current_bbox
                return True, [x1, y1, x2 - x1, y2 - y1]

            else:
                # IoU < threshold - об'єкт не знайдено за geometric matching

                self.lost_frames += 1

                if self.lost_frames < self.max_lost_frames:
                    # ========================================
                    # ФАЗА 2: ПОШУК-ОЧІКУВАННЯ з DIoU
                    # ========================================

                    # Очистити top candidates (вони були для Phase 1)
                    self.top_candidates = []

                    # Зберегти всі detections для візуалізації з інформацією про статус
                    self.search_candidates = []
                    for box in boxes:
                        box_xyxy = box.xyxy[0].cpu().numpy()
                        box_conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = box_xyxy
                        diou = self._compute_diou(self.last_valid_bbox, box_xyxy) if self.last_valid_bbox else -float('inf')

                        # Визначити статус для кожного候選
                        status = 'rejected'  # Default
                        if diou >= self.phase2_diou_threshold:
                            status = 'accepted'  # DIoU match
                        elif box_conf >= self.waiting_reinit_conf_threshold and diou >= self.waiting_reinit_diou_threshold:
                            status = 'accepted'  # Early reinit

                        self.search_candidates.append({
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'conf': box_conf,
                            'diou': diou,
                            'phase': 2,
                            'status': status
                        })

                    # Спробувати DIoU matching якщо увімкнено
                    phase2_diou_idx = -1
                    phase2_diou_value = -float('inf')

                    if self.phase2_diou_threshold < 1.0 and self.last_valid_bbox:
                        # DIoU увімкнено для Phase 2 (threshold < 1.0)
                        # Селекція: чистий max(DIoU) або (joint) max joint-балу серед
                        # кандидатів, що пройшли просторовий gate — gate незмінний.
                        use_joint_p2 = self.use_joint_score and self.joint_apply_phase2
                        lam_p2 = self._adaptive_lambda() if use_joint_p2 else None
                        best_p2_sel = -float('inf')
                        for idx, box in enumerate(boxes):
                            box_xyxy = box.xyxy[0].cpu().numpy()
                            diou = self._compute_diou(self.last_valid_bbox, box_xyxy)
                            if diou < self.phase2_diou_threshold:
                                continue
                            if use_joint_p2:
                                box_conf = float(box.conf[0].cpu().numpy())
                                sel = self._joint_score(box_conf, diou, lam_p2)
                            else:
                                sel = diou
                            if sel > best_p2_sel:
                                best_p2_sel = sel
                                phase2_diou_value = diou
                                phase2_diou_idx = idx

                    # Якщо знайдено детекцію за DIoU
                    if phase2_diou_idx >= 0:
                        best_box = boxes[phase2_diou_idx]
                        box_xyxy = best_box.xyxy[0].cpu().numpy()
                        box_conf = float(best_box.conf[0].cpu().numpy())

                        # Перевірка чи потрібна temporal validation для Phase 2
                        should_switch_immediately = False

                        if self.phase2_switch_validation_frames > 0:
                            # ===== VALIDATION MODE =====
                            # Створити інформацію про кандидата
                            current_candidate = {
                                'idx': phase2_diou_idx,
                                'bbox': box_xyxy,
                                'conf': box_conf,
                                'diou': phase2_diou_value
                            }

                            # Перевірити чи це той самий кандидат що і раніше
                            is_same_candidate = False
                            if self.phase2_pending_candidate is not None:
                                prev_bbox = self.phase2_pending_candidate['bbox']
                                curr_bbox = current_candidate['bbox']
                                candidate_iou = self._compute_iou(prev_bbox, curr_bbox)
                                # Вважаємо що це той самий кандидат якщо IoU > 0.5
                                is_same_candidate = candidate_iou > 0.5

                            if is_same_candidate:
                                # Продовжити валідацію того самого кандидата
                                self.phase2_pending_validation_count += 1
                                if self.verbose:
                                    print(f"   🔶 [PHASE 2] Pending валідація {self.phase2_pending_validation_count}/{self.phase2_switch_validation_frames}")
                                    print(f"      Кандидат: DIoU={current_candidate['diou']:.3f}, conf={current_candidate['conf']:.3f}")

                                # Перевірити чи досягнуто порогу валідації
                                if self.phase2_pending_validation_count >= self.phase2_switch_validation_frames:
                                    # Валідація успішна - перемкнутися!
                                    should_switch_immediately = True
                                    if self.verbose:
                                        print(f"   ✅ [PHASE 2] Pending валідація пройдена - перемикання на новий об'єкт")
                            else:
                                # Новий кандидат - почати валідацію спочатку
                                self.phase2_pending_candidate = current_candidate
                                self.phase2_pending_validation_count = 1
                                if self.verbose:
                                    print(f"   🆕 [PHASE 2] Новий pending кандидат виявлено")
                                    print(f"      DIoU={current_candidate['diou']:.3f}, conf={current_candidate['conf']:.3f}")
                                    print(f"      Початок валідації 1/{self.phase2_switch_validation_frames}")
                        else:
                            # ===== IMMEDIATE MODE (без validation) =====
                            should_switch_immediately = True

                        # Перемкнутися якщо потрібно
                        if should_switch_immediately:
                            # Оновити bbox (з Калманом якщо увімкнено)
                            detected_bbox_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]
                            if self.use_kalman and self.kalman is not None:
                                smoothed_bbox = self.kalman.update(detected_bbox_xywh)
                                x, y, w, h = smoothed_bbox
                                self.current_bbox = [x, y, x + w, y + h]
                            else:
                                self.current_bbox = box_xyxy.tolist()

                            self.last_valid_bbox = self.current_bbox
                            self.lost_frames = 0  # Reset counter
                            self.last_bbox_is_kalman_only = False  # Детекція знайдена - bbox валідовано
                            self.search_candidates = []  # Очистити кандидатів (Phase 2 - знайдено)

                            # ⭐ Скинути Phase 3 validation (Phase 2 commit перериває її)
                            if self.in_phase3_validation:
                                self.in_phase3_validation = False
                                self.phase3_validation_bbox = None
                                self.phase3_validation_consecutive_successes = 0
                                self.phase3_validation_failure_count = 0

                            # Скинути pending стан
                            self.phase2_pending_candidate = None
                            self.phase2_pending_validation_count = 0

                            # ⭐ Заповнити top_candidates для візуалізації на першому фреймі після відновлення
                            self.top_candidates = []
                            for idx, box in enumerate(boxes):
                                box_xyxy_tmp = box.xyxy[0].cpu().numpy()
                                curr_box_conf = float(box.conf[0].cpu().numpy())
                                diou_val = self._compute_diou(self.last_valid_bbox, box_xyxy_tmp)
                                x1, y1, x2, y2 = box_xyxy_tmp
                                is_best = (idx == phase2_diou_idx)

                                # Визначити статус кандидата
                                status = 'rejected'
                                if diou_val >= self.phase2_diou_threshold:
                                    status = 'accepted'
                                elif curr_box_conf >= self.waiting_reinit_conf_threshold and diou_val >= self.waiting_reinit_diou_threshold:
                                    status = 'accepted'

                                self.top_candidates.append({
                                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                                    'iou': diou_val,  # For Phase 2, use DIoU as IoU score
                                    'diou': diou_val,
                                    'conf': curr_box_conf,
                                    'is_best_match': is_best,
                                    'size_valid': True,
                                    'type': 'phase2',
                                    'phase': 2,
                                    'status': status
                                })

                            if self.verbose:
                                kalman_suffix = " + Kalman" if self.use_kalman else ""
                                print(f"✅ Кадр {self.frame_count}: [PHASE 2] DIoU Match{kalman_suffix} (IoU={best_iou:.3f}, DIoU={phase2_diou_value:.3f} >= {self.phase2_diou_threshold}, Conf={box_conf:.3f})")

                            # Зібрати VPE якщо час (або pending) та conf достатня
                            # Під час warmup збирати кожен кадр
                            should_collect_vpe = self.in_warmup or (self.frame_count % self.vpe_step == 0) or self.vpe_pending

                            if should_collect_vpe:
                                # Під час warmup використовувати мінімальний поріг (той що використовується для детекції)
                                if self.in_warmup:
                                    current_vpe_threshold = self._get_adaptive_conf()  # Той самий conf що для детекції
                                else:
                                    current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()

                                if box_conf >= current_vpe_threshold:
                                    if self.verbose:
                                        pending_msg = " (pending)" if self.vpe_pending else ""
                                        print(f"🔄 Кадр {self.frame_count}: Збір VPE{pending_msg} (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={self._get_vpe_count()}/{self.max_vpe})")
                                    self._collect_vpe(image, self.current_bbox, box_conf)
                                    self.vpe_pending = False
                                else:
                                    self.vpe_pending = True
                                    if self.verbose:
                                        print(f"⚠️  Кадр {self.frame_count}: VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={self._get_vpe_count()}/{self.max_vpe}), спроба на наступному кадрі")

                            x1, y1, x2, y2 = self.current_bbox
                            return True, [x1, y1, x2 - x1, y2 - y1]
                    else:
                        # Кандидат не знайдено або втрачено - скинути pending
                        if self.phase2_pending_candidate is not None:
                            if self.verbose:
                                print(f"   ❌ [PHASE 2] Pending скинуто (кандидат втрачено)")
                            self.phase2_pending_candidate = None
                            self.phase2_pending_validation_count = 0

                    # Перевірити чи є детекція з високою conf та прийнятним DIoU для дострокової реініціалізації
                    waiting_reinit_idx = -1
                    waiting_reinit_conf = 0
                    waiting_reinit_diou = -float('inf')

                    if self.waiting_reinit_conf_threshold < 1.0 and self.last_valid_bbox:
                        for idx, box in enumerate(boxes):
                            box_xyxy = box.xyxy[0].cpu().numpy()
                            box_conf = float(box.conf[0].cpu().numpy())
                            diou = self._compute_diou(self.last_valid_bbox, box_xyxy)

                            # Перевірити conf та DIoU пороги
                            if box_conf >= self.waiting_reinit_conf_threshold and diou >= self.waiting_reinit_diou_threshold:
                                if box_conf > waiting_reinit_conf:
                                    waiting_reinit_conf = box_conf
                                    waiting_reinit_idx = idx
                                    waiting_reinit_diou = diou

                    # Якщо знайдено високоякісну детекцію, реініціалізувати
                    if waiting_reinit_idx >= 0:
                        best_box = boxes[waiting_reinit_idx]
                        box_xyxy = best_box.xyxy[0].cpu().numpy()
                        box_conf = float(best_box.conf[0].cpu().numpy())

                        # Оновити bbox (з Калманом якщо увімкнено)
                        detected_bbox_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]
                        if self.use_kalman and self.kalman is not None:
                            smoothed_bbox = self.kalman.update(detected_bbox_xywh)
                            x, y, w, h = smoothed_bbox
                            self.current_bbox = [x, y, x + w, y + h]
                        else:
                            self.current_bbox = box_xyxy.tolist()

                        self.last_valid_bbox = self.current_bbox
                        self.lost_frames = 0  # Reset counter

                        # ⭐ Скинути Phase 3 validation (Phase 2 waiting_reinit commit перериває її)
                        if self.in_phase3_validation:
                            self.in_phase3_validation = False
                            self.phase3_validation_bbox = None
                            self.phase3_validation_consecutive_successes = 0
                            self.phase3_validation_failure_count = 0

                        # ⭐ Заповнити top_candidates для візуалізації на першому фреймі після відновлення
                        self.top_candidates = []
                        for idx, box in enumerate(boxes):
                            box_xyxy_tmp = box.xyxy[0].cpu().numpy()
                            curr_box_conf = float(box.conf[0].cpu().numpy())
                            diou_val = self._compute_diou(self.last_valid_bbox, box_xyxy_tmp)
                            x1, y1, x2, y2 = box_xyxy_tmp
                            is_best = (idx == waiting_reinit_idx)

                            # Визначити статус кандидата
                            status = 'rejected'
                            if curr_box_conf >= self.waiting_reinit_conf_threshold and diou_val >= self.waiting_reinit_diou_threshold:
                                status = 'accepted'

                            self.top_candidates.append({
                                'bbox': [x1, y1, x2 - x1, y2 - y1],
                                'iou': diou_val,
                                'diou': diou_val,
                                'conf': curr_box_conf,
                                'is_best_match': is_best,
                                'size_valid': True,
                                'type': 'phase2',
                                'phase': 2,
                                'status': status
                            })

                        if self.verbose:
                            kalman_suffix = " + Kalman" if self.use_kalman else ""
                            print(f"🔄 Кадр {self.frame_count}: [PHASE 2] Early Reinit{kalman_suffix} (conf={box_conf:.3f} >= {self.waiting_reinit_conf_threshold}, DIoU={waiting_reinit_diou:.3f} >= {self.waiting_reinit_diou_threshold})")

                        # Зібрати VPE для нового bbox (якщо conf достатня)
                        # Під час warmup використовувати мінімальний поріг
                        if self.in_warmup:
                            current_vpe_threshold = self._get_adaptive_conf()
                        else:
                            current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()

                        if box_conf >= current_vpe_threshold:
                            if self.verbose:
                                print(f"   📥 Збір VPE для нового bbox (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={self._get_vpe_count()}/{self.max_vpe})")
                            self._collect_vpe(image, self.current_bbox, box_conf)
                            self.vpe_pending = False
                        else:
                            self.vpe_pending = True
                            if self.verbose:
                                print(f"   ⚠️  VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={self._get_vpe_count()}/{self.max_vpe}), спроба на наступному кадрі")

                        x1, y1, x2, y2 = self.current_bbox
                        return True, [x1, y1, x2 - x1, y2 - y1]

                    # DIoU не спрацював або вимкнено - режим очікування
                    if self.verbose:
                        diou_msg = f", DIoU={phase2_diou_value:.3f} < {self.phase2_diou_threshold}" if self.phase2_diou_threshold < 1.0 else ""
                        kalman_msg = " - using Kalman prediction" if (self.use_kalman and kalman_prediction is not None) else ""
                        print(f"⚠️  Кадр {self.frame_count}: [PHASE 2] Waiting (IoU={best_iou:.3f} < {self.iou_threshold}{diou_msg}, lost={self.lost_frames}/{self.max_lost_frames}){kalman_msg}")

                    # Якщо є Калман прогноз, повертаємо його замість False
                    # Це дозволяє продовжити трекінг з прогнозом під час короткочасної втрати
                    if self.use_kalman and kalman_prediction is not None:
                        # Оновити current_bbox на Kalman prediction для неперервності траєкторії
                        x, y, w, h = kalman_prediction  # xywh
                        self.current_bbox = [x, y, x + w, y + h]  # convert to xyxy
                        self.last_bbox_is_kalman_only = True  # Відмітити, що це не валідовано детекціями
                        # ⭐ Зберегти останню Phase 2 Kalman предикцію для Phase 3 ref
                        self.phase3_kalman_exit_bbox = self.current_bbox[:]
                        return True, kalman_prediction
                    else:
                        # Не оновлюємо bbox, повертаємо False (втрачений для евалюації)
                        # last_valid_bbox залишається для можливої реідентифікації
                        return False, None

                else:
                    # ========================================
                    # ФАЗА 3: РЕІДЕНТИФІКАЦІЯ
                    # ========================================

                    # ⭐ Phase 3 Validation Gate: перевірка pending кандидата
                    if self.in_phase3_validation and self.phase3_validation_bbox is not None:
                        # Шукаємо кандидата в поточних детекціях
                        best_val_iou = 0.0
                        best_val_box = None
                        best_val_conf = 0.0
                        if boxes is not None:
                            for box in boxes:
                                box_xyxy = box.xyxy[0].cpu().numpy()
                                val_iou = self._compute_iou(self.phase3_validation_bbox, box_xyxy)
                                if val_iou > best_val_iou:
                                    best_val_iou = val_iou
                                    best_val_box = box_xyxy
                                    best_val_conf = float(box.conf[0].cpu().numpy())

                        if best_val_iou >= self.iou_threshold:
                            # Кандидат видимий — оновити його позицію та лічильник
                            self.phase3_validation_bbox = best_val_box.tolist()
                            self.phase3_validation_consecutive_successes += 1
                            if self.verbose:
                                print(f"📊 Кадр {self.frame_count}: [PHASE 3 VAL] {self.phase3_validation_consecutive_successes}/{self.phase3_redetection_validation_frames}, IoU={best_val_iou:.3f}, conf={best_val_conf:.3f}")

                            if self.phase3_validation_consecutive_successes >= self.phase3_redetection_validation_frames:
                                # COMMIT: validation пройдена, переходимо до Phase 1
                                val_bbox = best_val_box
                                detected_bbox_xywh = [val_bbox[0], val_bbox[1], val_bbox[2] - val_bbox[0], val_bbox[3] - val_bbox[1]]
                                if self.use_kalman and self.kalman is not None:
                                    smoothed = self.kalman.update(detected_bbox_xywh)
                                    x, y, w, h = smoothed
                                    self.current_bbox = [x, y, x + w, y + h]
                                else:
                                    self.current_bbox = best_val_box.tolist()
                                self.last_valid_bbox = self.current_bbox
                                self.lost_frames = 0
                                self.last_bbox_is_kalman_only = False
                                self.in_phase3_validation = False
                                self.phase3_validation_consecutive_successes = 0
                                self.phase3_validation_failure_count = 0
                                self.phase3_validation_bbox = None
                                self.vpe_pending = True  # Зібрати VPE після підтвердження
                                if self.verbose:
                                    print(f"✅ Кадр {self.frame_count}: [PHASE 3 VAL] PASSED — об'єкт підтверджено, повертаємось до Phase 1")
                                x1, y1, x2, y2 = self.current_bbox
                                return True, [x1, y1, x2 - x1, y2 - y1]
                            else:
                                return False, None  # Продовжуємо валідацію
                        else:
                            # Кандидат не видимий — невдача: скасувати validation негайно
                            # (потрібно строго N ПОСЛІДОВНИХ кадрів, будь-яка невдача скидає лічильник)
                            if self.verbose:
                                print(f"❌ Кадр {self.frame_count}: [PHASE 3 VAL] невдача (consecutive={self.phase3_validation_consecutive_successes}) — validation скасовано, Phase 3 шукає знову")
                            self.in_phase3_validation = False
                            self.phase3_validation_bbox = None
                            self.phase3_validation_consecutive_successes = 0
                            self.phase3_validation_failure_count = 0
                            # Fall through до нормального Phase 3 search (шукаємо кандидата знову)

                    # Очистити попередні відкинуті кандидати
                    self.rejected_candidates = []

                    # Обчислити адаптивний DIoU поріг для Phase 3
                    phase3_current_threshold = self._get_adaptive_diou_threshold()

                    # ⭐ Визначити reference bbox для Phase 3 порівнянь (залежно від phase3_ref_mode):
                    if self.phase3_ref_mode == 'kalman_phase3' and self.use_kalman and kalman_prediction is not None:
                        # Поточна Kalman предикція у Phase 3 (найсвіжіша, але може дрейфувати)
                        kx, ky, kw, kh = kalman_prediction
                        phase3_ref_bbox = [kx, ky, kx + kw, ky + kh]  # xywh → xyxy
                        ref_src_log = "kalman_phase3"
                    elif self.phase3_ref_mode == 'kalman_phase2' and self.phase3_kalman_exit_bbox is not None:
                        # Остання Kalman предикція з Phase 2 (стабільніша ніж phase3, враховує рух)
                        phase3_ref_bbox = self.phase3_kalman_exit_bbox
                        ref_src_log = "kalman_phase2"
                    else:
                        # 'last_valid' або fallback: остання детектована позиція
                        phase3_ref_bbox = self.last_valid_bbox
                        ref_src_log = "last_valid" + ("" if self.phase3_ref_mode == 'last_valid' else " (fallback)")
                    self.phase3_reference_bbox = phase3_ref_bbox  # Зберегти для візуалізації
                    if self.verbose:
                        rx1, ry1, rx2, ry2 = phase3_ref_bbox
                        print(f"   📍 [PHASE 3] ref_bbox [{ref_src_log}]: [{rx1:.1f},{ry1:.1f},{rx2:.1f},{ry2:.1f}]")

                    # Зберегти всі detections для візуалізації з інформацією про статус
                    self.search_candidates = []
                    for box in boxes:
                        box_xyxy = box.xyxy[0].cpu().numpy()
                        box_conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = box_xyxy
                        diou = self._compute_diou(phase3_ref_bbox, box_xyxy) if phase3_ref_bbox else -float('inf')

                        # Визначити статус для кожного кандидата
                        status = 'rejected'  # Default
                        if box_conf >= self.reinit_conf_threshold:
                            # High-conf reinit: також перевіряємо DIoU якщо заданий
                            diou_ok = (self.reinit_diou_threshold <= -1.0 or
                                       not self.last_valid_bbox or
                                       diou >= self.reinit_diou_threshold)
                            if diou_ok:
                                status = 'accepted'  # High-conf + DIoU reinit
                        if status == 'rejected' and diou >= phase3_current_threshold:
                            status = 'accepted'  # DIoU-based reinit

                        self.search_candidates.append({
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'conf': box_conf,
                            'diou': diou,
                            'phase': 3,
                            'status': status
                        })

                    # ⭐ Joint DIoU×conf score (за прапорцем use_joint_score):
                    # єдиний неперервний бал замість окремих conf/DIoU reinit-гейтів
                    # + вибору max(conf). Коротко замикає легасі-шляхи нижче.
                    if self.use_joint_score and phase3_ref_bbox:
                        lam = self._adaptive_lambda()
                        joint_eligible = []
                        for idx, box in enumerate(boxes):
                            box_xyxy = box.xyxy[0].cpu().numpy()
                            box_conf = float(box.conf[0].cpu().numpy())
                            diou = self._compute_diou(phase3_ref_bbox, box_xyxy)
                            if diou < self.reinit_diou_floor:   # anti-teleport sanity floor
                                x1, y1, x2, y2 = box_xyxy
                                self.rejected_candidates.append({
                                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                                    'conf': box_conf, 'diou': diou})
                                continue
                            joint_eligible.append({
                                'idx': idx, 'conf': box_conf, 'xyxy': box_xyxy,
                                'diou': diou,
                                'score': self._joint_score(box_conf, diou, lam)})

                        if self.verbose and joint_eligible:
                            print(f"   🔍 Фаза 3 [joint λ={lam:.2f}, {self.joint_mode}, thr={self.reinit_joint_threshold}]: {len(joint_eligible)} кандидатів:")
                            for c in sorted(joint_eligible, key=lambda x: -x['score']):
                                print(f"      score={c['score']:.3f} | conf={c['conf']:.3f}, DIoU={c['diou']:.3f}")

                        if joint_eligible:
                            best = max(joint_eligible, key=lambda c: c['score'])
                            if best['score'] >= self.reinit_joint_threshold:
                                return self._commit_phase3_reid(
                                    best['xyxy'], best['conf'],
                                    tag=f"Joint(score={best['score']:.3f},DIoU={best['diou']:.3f})")
                        if self.verbose:
                            best_s = max((c['score'] for c in joint_eligible), default=float('nan'))
                            print(f"❌ Кадр {self.frame_count}: [PHASE 3] Joint Re-ID failed (best score {best_s:.3f} < {self.reinit_joint_threshold})")
                        return False, None

                    # Спочатку перевіряємо чи є детекція з високою conf (автоматична реініціалізація)
                    high_conf_candidate_idx = -1
                    high_conf_candidate_conf = 0

                    if self.reinit_conf_threshold < 1.0:
                        high_conf_eligible = []
                        for idx, box in enumerate(boxes):
                            box_conf = float(box.conf[0].cpu().numpy())
                            if box_conf >= self.reinit_conf_threshold:
                                box_xyxy_temp = box.xyxy[0].cpu().numpy()
                                # ⭐ Також перевіряємо DIoU якщо reinit_diou_threshold заданий
                                if self.reinit_diou_threshold > -1.0 and phase3_ref_bbox:
                                    diou_temp = self._compute_diou(phase3_ref_bbox, box_xyxy_temp)
                                    if diou_temp < self.reinit_diou_threshold:
                                        continue  # DIoU занадто малий — пропустити
                                high_conf_eligible.append({'idx': idx, 'conf': box_conf,
                                                           'xyxy': box_xyxy_temp})
                        if high_conf_eligible:
                            chosen = self._select_phase3_candidate(image, high_conf_eligible,
                                                                   tag="high-conf")
                            high_conf_candidate_idx = chosen['idx']
                            high_conf_candidate_conf = chosen['conf']

                    # Якщо знайдено високоякісну детекцію, використати її
                    if high_conf_candidate_idx >= 0:
                        best_box = boxes[high_conf_candidate_idx]
                        box_xyxy = best_box.xyxy[0].cpu().numpy()
                        box_conf = float(best_box.conf[0].cpu().numpy())

                        if self.phase3_redetection_validation_frames > 0:
                            # Зберегти кандидата для validation — НЕ оновлювати last_valid_bbox/lost_frames
                            self.in_phase3_validation = True
                            self.phase3_validation_consecutive_successes = 0
                            self.phase3_validation_failure_count = 0
                            self.phase3_validation_bbox = box_xyxy.tolist()
                            if self.phase3_vpe_freeze_frames > 0:
                                self.phase3_vpe_freeze_counter = self.phase3_vpe_freeze_frames
                            if self.verbose:
                                print(f"🔍 Кадр {self.frame_count}: [PHASE 3] High-Conf кандидат (conf={box_conf:.3f}), починаємо validation на {self.phase3_redetection_validation_frames} кадрів")
                            return False, None
                        else:
                            # Validation вимкнена — миттєвий commit
                            detected_bbox_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]
                            if self.use_kalman and self.kalman is not None:
                                smoothed_bbox = self.kalman.update(detected_bbox_xywh)
                                x, y, w, h = smoothed_bbox
                                self.current_bbox = [x, y, x + w, y + h]
                            else:
                                self.current_bbox = box_xyxy.tolist()
                            self.last_valid_bbox = self.current_bbox
                            self.lost_frames = 0
                            self.last_bbox_is_kalman_only = False
                            self.search_candidates = []
                            if self.verbose:
                                print(f"🔄 Кадр {self.frame_count}: [PHASE 3] High-Conf Re-ID (conf={box_conf:.3f} >= {self.reinit_conf_threshold})")
                            x1, y1, x2, y2 = self.current_bbox
                            return True, [x1, y1, x2 - x1, y2 - y1]

                    # Якщо встановлено reinit_diou_threshold, фільтруємо кандидатів за DIoU
                    reinit_candidate_idx = -1
                    reinit_candidate_conf = 0
                    reinit_candidate_diou = -float('inf')

                    if self.reinit_diou_threshold > -1.0 and phase3_ref_bbox:
                        # Обчислити адаптивний threshold на основі lost_frames
                        current_threshold = self._get_adaptive_diou_threshold()

                        # Фільтрація за DIoU від phase3_ref_bbox (Kalman або last_valid_bbox)
                        all_candidates = []  # Всі кандидати для verbose виводу
                        reinit_eligible = []  # Кандидати, що пройшли DIoU-гейт

                        for idx, box in enumerate(boxes):
                            box_xyxy = box.xyxy[0].cpu().numpy()
                            box_conf = float(box.conf[0].cpu().numpy())

                            # Обчислити DIoU з phase3_ref_bbox
                            diou = self._compute_diou(phase3_ref_bbox, box_xyxy)

                            # Зберегти кандидата
                            candidate_info = {
                                'bbox': box_xyxy.tolist(),
                                'conf': box_conf,
                                'diou': diou,
                                'accepted': diou >= current_threshold
                            }
                            all_candidates.append(candidate_info)

                            # Перевірити чи в межах threshold
                            if diou >= current_threshold:
                                reinit_eligible.append({'idx': idx, 'conf': box_conf,
                                                        'xyxy': box_xyxy, 'diou': diou})
                            else:
                                # Зберегти відкинутого кандидата для візуалізації
                                x1, y1, x2, y2 = box_xyxy
                                self.rejected_candidates.append({
                                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                                    'conf': box_conf,
                                    'diou': diou
                                })

                        # Verbose вивід про всіх кандидатів
                        if self.verbose and len(all_candidates) > 0:
                            # Показати поточний threshold (з адаптацією якщо є)
                            frames_in_phase3 = self.lost_frames - self.max_lost_frames
                            threshold_info = f"DIoU >= {current_threshold:.3f}"
                            if self.reinit_diou_max < self.reinit_diou_threshold:
                                threshold_info += f" (adaptive: frame {frames_in_phase3}/{self.reinit_adaptive_rate})"

                            ref_src = "Kalman" if (self.use_kalman and kalman_prediction is not None) else "last_valid"
                            print(f"   🔍 Фаза 3: {len(all_candidates)} кандидатів (threshold {threshold_info}, ref={ref_src}):")
                            for i, cand in enumerate(all_candidates):
                                status = "✅ ПРИЙНЯТО" if cand['accepted'] else "❌ ВІДКИНУТО"
                                print(f"      #{i+1}: {status} | conf={cand['conf']:.3f}, DIoU={cand['diou']:.3f}")

                        if reinit_eligible:
                            chosen = self._select_phase3_candidate(image, reinit_eligible, tag="DIoU")
                            reinit_candidate_idx = chosen['idx']
                            reinit_candidate_conf = chosen['conf']
                            reinit_candidate_diou = chosen['diou']

                        if reinit_candidate_idx >= 0:
                            best_box = boxes[reinit_candidate_idx]
                            box_xyxy = best_box.xyxy[0].cpu().numpy()
                            box_conf = float(best_box.conf[0].cpu().numpy())

                            if self.phase3_redetection_validation_frames > 0:
                                # Зберегти кандидата для validation — НЕ оновлювати last_valid_bbox/lost_frames
                                self.in_phase3_validation = True
                                self.phase3_validation_consecutive_successes = 0
                                self.phase3_validation_failure_count = 0
                                self.phase3_validation_bbox = box_xyxy.tolist()
                                if self.phase3_vpe_freeze_frames > 0:
                                    self.phase3_vpe_freeze_counter = self.phase3_vpe_freeze_frames
                                if self.verbose:
                                    print(f"🔍 Кадр {self.frame_count}: [PHASE 3] DIoU кандидат (conf={box_conf:.3f}, DIoU={reinit_candidate_diou:.3f}), validation на {self.phase3_redetection_validation_frames} кадрів")
                                return False, None
                            else:
                                # Validation вимкнена — миттєвий commit
                                detected_bbox_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]
                                if self.use_kalman and self.kalman is not None:
                                    smoothed_bbox = self.kalman.update(detected_bbox_xywh)
                                    x, y, w, h = smoothed_bbox
                                    self.current_bbox = [x, y, x + w, y + h]
                                else:
                                    self.current_bbox = box_xyxy.tolist()
                                self.last_valid_bbox = self.current_bbox
                                self.lost_frames = 0
                                self.last_bbox_is_kalman_only = False
                                self.search_candidates = []
                                if self.verbose:
                                    print(f"🔄 Кадр {self.frame_count}: [PHASE 3] Re-ID (conf={box_conf:.3f}, DIoU={reinit_candidate_diou:.3f})")
                                x1, y1, x2, y2 = self.current_bbox
                                return True, [x1, y1, x2 - x1, y2 - y1]
                        else:
                            # Всі detections з низьким DIoU
                            if self.verbose:
                                print(f"❌ Кадр {self.frame_count}: [PHASE 3] Re-ID failed (всі кандидати: DIoU < {current_threshold:.3f})")
                            return False, None

                    else:
                        # Немає обмеження на DIoU — max(conf) або appearance-зважений вибір
                        if best_conf_idx >= 0 and self.phase3_appearance_weight > 0.0:
                            fallback_eligible = [
                                {'idx': idx,
                                 'conf': float(box.conf[0].cpu().numpy()),
                                 'xyxy': box.xyxy[0].cpu().numpy()}
                                for idx, box in enumerate(boxes)
                            ]
                            chosen = self._select_phase3_candidate(image, fallback_eligible,
                                                                   tag="max-conf")
                            best_conf_idx = chosen['idx']
                        if best_conf_idx >= 0:
                            best_box = boxes[best_conf_idx]
                            box_xyxy = best_box.xyxy[0].cpu().numpy()
                            box_conf = float(best_box.conf[0].cpu().numpy())

                            if self.phase3_redetection_validation_frames > 0:
                                # Зберегти кандидата для validation — НЕ оновлювати last_valid_bbox/lost_frames
                                self.in_phase3_validation = True
                                self.phase3_validation_consecutive_successes = 0
                                self.phase3_validation_failure_count = 0
                                self.phase3_validation_bbox = box_xyxy.tolist()
                                if self.phase3_vpe_freeze_frames > 0:
                                    self.phase3_vpe_freeze_counter = self.phase3_vpe_freeze_frames
                                if self.verbose:
                                    print(f"🔍 Кадр {self.frame_count}: [PHASE 3] max_conf кандидат (conf={box_conf:.3f}), validation на {self.phase3_redetection_validation_frames} кадрів")
                                return False, None
                            else:
                                # Validation вимкнена — миттєвий commit
                                detected_bbox_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]
                                if self.use_kalman and self.kalman is not None:
                                    smoothed_bbox = self.kalman.update(detected_bbox_xywh)
                                    x, y, w, h = smoothed_bbox
                                    self.current_bbox = [x, y, x + w, y + h]
                                else:
                                    self.current_bbox = box_xyxy.tolist()
                                self.last_valid_bbox = self.current_bbox
                                self.lost_frames = 0
                                self.last_bbox_is_kalman_only = False
                                if self.verbose:
                                    print(f"🔄 Кадр {self.frame_count}: [PHASE 3] Re-ID (max_conf={box_conf:.3f})")
                                x1, y1, x2, y2 = self.current_bbox
                                return True, [x1, y1, x2 - x1, y2 - y1]
                        else:
                            # Немає detections взагалі
                            if self.verbose:
                                print(f"❌ Кадр {self.frame_count}: [PHASE 3] Re-ID failed (no detections)")
                            return False, None

        except Exception as e:
            if self.verbose:
                print(f"❌ Помилка update: {e}")
            return False, None

    def _get_vpe_predictor(self):
        """
        Лінива ініціалізація виділеного VP-предиктора для видобування VPE.

        Ділить той самий nn.Module з основною моделлю (окремих ваг не вантажиться)
        і не чіпає self.model.predictor, тож детекційний предиктор не
        перестворюється на кожному кадрі збору VPE.
        """
        if self._vpe_predictor is None:
            self._vpe_predictor = YOLOEVPSegPredictor(
                overrides=dict(
                    task="segment", mode="predict", model=self.model_path,
                    imgsz=self.imgsz, conf=0.001, save=False, batch=1,
                    verbose=False, device=self.device,
                    # rect=True — як у предиктора, що створювався через YOLOE.predict
                    # (успадковував rect із overrides обгортки); без нього letterbox
                    # квадратний і VPE чисельно інший
                    rect=True,
                ),
                _callbacks=None,
            )
            self._vpe_predictor.setup_model(self.model.model, verbose=False)
        return self._vpe_predictor

    # --- Phase 3: appearance-зважений вибір кандидата реініціалізації ---------
    def _get_appearance_ref_vpe(self, mode: Optional[str] = None):
        """Референсний VPE пам'яті ('anchor' | 'aggregated'):
        'anchor' — VPE першого кадру (лише dual memory); 'aggregated' — поточний
        агрегат пам'яті, що еволюціонує разом із виглядом об'єкта (за суттєвої
        зміни вигляду, напр. дрон у польоті на LaSOT, надійніший за anchor).
        mode=None -> phase3_appearance_ref (для Phase 3 вибору)."""
        mode = mode or self.phase3_appearance_ref
        if mode == 'anchor' and self.dual_memory is not None:
            anchor = self.dual_memory.get_anchor_vpe()
            if anchor is not None:
                return anchor
        return self.aggregated_vpe

    def _encode_boxes_vpe_batch(self, image: np.ndarray, xyxy_list):
        """L2-нормовані VPE для списку рамок xyxy за ОДИН forward
        (унікальний cls на рамку -> SAVPE дає окремий ембеддинг на кожну).
        Повертає тензор (N, D) або None."""
        if xyxy_list is None or len(xyxy_list) == 0:
            return None
        H, W = image.shape[:2]
        clipped = []
        for b in xyxy_list:
            x1, y1, x2, y2 = [float(v) for v in b]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            clipped.append([x1, y1, max(x1 + 1.0, x2), max(y1 + 1.0, y2)])
        try:
            vp = self._get_vpe_predictor()
            vp.set_prompts(dict(bboxes=np.array(clipped, dtype=np.float32),
                                cls=np.arange(len(clipped), dtype=np.int64)))
            with torch.no_grad():
                vpe = vp.get_vpe(image)  # (1, N, D)
            feats = vpe.reshape(len(clipped), -1).float()
            return F.normalize(feats, dim=1)
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Phase 3 batch VPE error: {e}")
            return None

    def _select_phase3_candidate(self, image: np.ndarray, eligible: List[Dict], tag: str = "") -> Dict:
        """Вибір кандидата реініціалізації серед eligible=[{idx, conf, xyxy, ...}].
        За phase3_appearance_weight == 0 — max(conf) (поведінка без змін). Інакше
        score = (1-w)*conf + w*sim, де sim — косинус VPE кандидата з референсом
        пам'яті, масштабований у [0, 1]; VPE кандидатів — одним батчованим forward."""
        w = self.phase3_appearance_weight
        if w <= 0.0 or len(eligible) == 1:
            return max(eligible, key=lambda c: c['conf'])
        ref = self._get_appearance_ref_vpe()
        if ref is None or not isinstance(ref, torch.Tensor):
            return max(eligible, key=lambda c: c['conf'])
        feats = self._encode_boxes_vpe_batch(image, [c['xyxy'] for c in eligible])
        if feats is None:
            return max(eligible, key=lambda c: c['conf'])
        ref_v = ref.flatten().float().to(feats.device)
        ref_v = ref_v / (ref_v.norm() + 1e-8)
        sims = ((feats @ ref_v + 1.0) / 2.0).tolist()  # [-1..1] -> [0..1]
        best, best_score = None, -float('inf')
        for c, s in zip(eligible, sims):
            c['sim'] = float(s)
            c['score'] = (1.0 - w) * c['conf'] + w * c['sim']
            if c['score'] > best_score:
                best_score, best = c['score'], c
        if self.verbose:
            print(f"   🎯 [PHASE 3 {tag}] appearance-вибір (w={w}, ref={self.phase3_appearance_ref}):")
            for c in sorted(eligible, key=lambda x: -x['score']):
                mark = " ← ОБРАНО" if c is best else ""
                print(f"      idx={c['idx']}: conf={c['conf']:.3f} sim={c['sim']:.3f} score={c['score']:.3f}{mark}")
        return best

    def _collect_vpe(self, image: np.ndarray, bbox: list, conf: float = 0.5):
        """
        Зібрати VPE з поточного кадру

        Args:
            image: Поточний кадр
            bbox: Bbox [x1, y1, x2, y2]
            conf: Detection confidence для VPE quality assessment [0-1]
        """
        try:
            # Створити visual prompts (завжди використовуємо cls=0 для VP моделей)
            visual_prompts = dict(
                bboxes=np.array([bbox]),
                cls=np.array([0]),
            )

            # Виділений VPE-предиктор: ОДИН forward (get_vpe) замість двох
            # (раніше повний model.predict запускався лише щоб створити predictor,
            #  а його результат не використовувався)
            vp = self._get_vpe_predictor()
            vp.set_prompts(visual_prompts)
            vpe = vp.get_vpe(image)

            # Validate VPE
            if vpe is None:
                if self.verbose:
                    print(f"   ⚠️  VPE is None, skipping collection")
                return

            if self.verbose:
                print(f"   📊 VPE shape: {vpe.shape}")

            # Гейт пам'яті: відхилити VPE, несхожий на референс пам'яті
            # (захист від забруднення пам'яті після дрейфу на дистрактор)
            if self.vpe_gate_threshold > 0.0:
                gate_ref = self._get_appearance_ref_vpe(self.vpe_gate_ref)
                if gate_ref is not None:
                    gate_sim = self._calculate_cosine_similarity(gate_ref, vpe)
                    if gate_sim < self.vpe_gate_threshold:
                        self.vpe_gate_rejected += 1
                        if self.verbose:
                            print(f"   🚫 VPE відхилено гейтом (sim={gate_sim:.3f} < "
                                  f"{self.vpe_gate_threshold}, ref={self.vpe_gate_ref})")
                        return
                    self.vpe_gate_accepted += 1

            # Додати до dual memory або simple deque
            if self.use_dual_memory_vpe:
                # Використовувати conf від detection (SAMURAI affinity або detection conf)
                self.dual_memory.add_vpe(vpe, conf, self.frame_count)

                if self.verbose:
                    stats = self.dual_memory.get_stats()
                    print(f"   📥 Зібрано VPE (conf={conf:.3f}): {self.dual_memory}")
            else:
                self.vpe_list.append(vpe)
                if self.verbose:
                    print(f"   📥 Зібрано VPE #{self._get_vpe_count()}")

            # Агрегувати VPE
            self._aggregate_vpe()

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Помилка збору VPE: {e}")

    def _get_vpe_count(self) -> int:
        """
        Отримати кількість зібраних VPE (dual memory або simple deque)

        Returns:
            int - кількість VPE
        """
        if self.use_dual_memory_vpe:
            return self.dual_memory.total_vpe_collected
        else:
            return len(self.vpe_list) if self.vpe_list else 0

    def _aggregate_vpe(self):
        """
        Агрегувати всі зібрані VPE
        """
        try:
            if self.use_dual_memory_vpe:
                # Use dual memory aggregation
                self.aggregated_vpe = self.dual_memory.get_aggregated_vpe()
                self._vpe_version += 1

                if self.verbose and self.aggregated_vpe is not None:
                    stats = self.dual_memory.get_stats()
                    print(f"   🔄 Dual Memory агрегація: LT={stats['long_term_count']}, ST={stats['short_term_count']}")
            else:
                # Use simple mean aggregation
                if self._get_vpe_count() == 0:
                    return

                # Об'єднати та усереднити: [1,1,D] + [1,1,D] = [1,N,D] -> [1,1,D]
                vpe_tensor = torch.cat(list(self.vpe_list), dim=1)
                self.aggregated_vpe = vpe_tensor.mean(dim=1, keepdim=True)

                # Нормалізувати
                self.aggregated_vpe = F.normalize(self.aggregated_vpe, p=2, dim=-1)
                self._vpe_version += 1

                if self.verbose:
                    print(f"   🔄 Агреговано {self._get_vpe_count()} VPE, shape={self.aggregated_vpe.shape}")

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Помилка агрегації: {e}")

    def _adaptive_lambda(self) -> float:
        """
        Вага просторової близькості (λ) для Phase 3 joint-score.

        Спадає від joint_lam_start (щойно загубився — довіряй позиції) до
        joint_lam_min (давно загублений — об'єкт міг переміститись, довіряй
        впевненості) за reinit_adaptive_rate кадрів у Фазі 3.
        """
        if self.lost_frames < self.max_lost_frames:
            return self.joint_lam_start
        frames_in_phase3 = self.lost_frames - self.max_lost_frames
        progress = min(1.0, frames_in_phase3 / max(1, self.reinit_adaptive_rate))
        return self.joint_lam_start * (1.0 - progress) + self.joint_lam_min * progress

    def _joint_score(self, conf: float, diou: float, lam: float) -> float:
        """
        Об'єднати впевненість і DIoU-близькість в один бал у [0,1].

        diou відображається [-1,1] -> [0,1] як (diou+1)/2 (та сама шкала, що й
        у _calculate_cosine_similarity). joint_mode:
          'geometric'  — conf^(1-λ)·diou_norm^λ (AND-подібний: обидва мають бути пристойні)
          'arithmetic' — (1-λ)·conf + λ·diou_norm (компенсація: висока conf рятує середній DIoU)
        """
        diou_norm = (diou + 1.0) / 2.0
        if self.joint_mode == 'arithmetic':
            return (1.0 - lam) * conf + lam * diou_norm
        return (max(conf, 1e-6) ** (1.0 - lam)) * (max(diou_norm, 1e-6) ** lam)

    def _commit_phase3_reid(self, box_xyxy, box_conf: float, tag: str = ""):
        """
        Застосувати (або поставити на validation) вибраний Phase 3 re-ID кандидат.

        Спільна commit-логіка (validation vs миттєвий commit + Kalman-згладжування),
        винесена щоб joint-score гілка не дублювала існуючі шляхи.
        """
        box_xyxy = [float(v) for v in box_xyxy]
        if self.phase3_redetection_validation_frames > 0:
            self.in_phase3_validation = True
            self.phase3_validation_consecutive_successes = 0
            self.phase3_validation_failure_count = 0
            self.phase3_validation_bbox = list(box_xyxy)
            if self.phase3_vpe_freeze_frames > 0:
                self.phase3_vpe_freeze_counter = self.phase3_vpe_freeze_frames
            if self.verbose:
                print(f"🔍 Кадр {self.frame_count}: [PHASE 3] {tag} кандидат (conf={box_conf:.3f}), validation на {self.phase3_redetection_validation_frames} кадрів")
            return False, None
        detected_bbox_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]
        if self.use_kalman and self.kalman is not None:
            smoothed_bbox = self.kalman.update(detected_bbox_xywh)
            x, y, w, h = smoothed_bbox
            self.current_bbox = [x, y, x + w, y + h]
        else:
            self.current_bbox = list(box_xyxy)
        self.last_valid_bbox = self.current_bbox
        self.lost_frames = 0
        self.last_bbox_is_kalman_only = False
        self.search_candidates = []
        if self.verbose:
            print(f"🔄 Кадр {self.frame_count}: [PHASE 3] {tag} Re-ID (conf={box_conf:.3f})")
        x1, y1, x2, y2 = self.current_bbox
        return True, [x1, y1, x2 - x1, y2 - y1]

    def _get_adaptive_diou_threshold(self) -> float:
        """
        Обчислити адаптивний DIoU threshold на основі кількості втрачених кадрів

        Returns:
            float - поточний DIoU threshold
        """
        # Якщо адаптація вимкнена або ще не в Фазі 3
        if self.reinit_diou_threshold <= -1.0 or self.lost_frames < self.max_lost_frames:
            return self.reinit_diou_threshold

        # Якщо адаптація вимкнена (max >= threshold, тобто не менш строге)
        if self.reinit_diou_max >= self.reinit_diou_threshold:
            return self.reinit_diou_threshold

        # Кількість кадрів після початку Фази 3
        frames_in_phase3 = self.lost_frames - self.max_lost_frames

        # Лінійна інтерполяція від reinit_diou_threshold до reinit_diou_max
        # Зменшуємо threshold (більш негативний = менш строгий)
        progress = min(1.0, frames_in_phase3 / self.reinit_adaptive_rate)
        current_threshold = self.reinit_diou_threshold + (self.reinit_diou_max - self.reinit_diou_threshold) * progress

        return current_threshold

    def _get_adaptive_conf(self) -> float:
        """
        Обчислити адаптивний conf threshold на основі кількості зібраних VPE

        Логіка: чим більше VPE зібрано, тим вищі вимоги до conf детекцій,
        щоб відфільтрувати низькоякісні спрацювання.

        Returns:
            float - поточний conf threshold
        """
        # Якщо адаптація вимкнена (max <= threshold, тобто не більш строге)
        if self.conf_max <= self.conf:
            return self.conf

        # Кількість зібраних VPE
        num_vpe = self._get_vpe_count()

        # Якщо ще немає VPE, використовуємо початковий threshold
        if num_vpe == 0:
            return self.conf

        # Лінійна інтерполяція від conf до conf_max
        # Збільшуємо threshold (більший = більш строгий)
        progress = min(1.0, num_vpe / self.conf_adaptive_rate)
        current_conf = self.conf + (self.conf_max - self.conf) * progress

        return current_conf

    def _get_adaptive_vpe_conf_threshold(self) -> float:
        """
        Обчислити адаптивний VPE conf threshold на основі кількості зібраних VPE

        Логіка: чим більше VPE зібрано, тим вищі вимоги до conf нових VPE,
        щоб не додавати нерелевантні низькоякісні детекції.

        Returns:
            float - поточний VPE conf threshold
        """
        # Якщо адаптація вимкнена (max <= threshold, тобто не більш строге)
        if self.vpe_conf_max <= self.vpe_conf_threshold:
            return self.vpe_conf_threshold

        # Кількість зібраних VPE
        num_vpe = self._get_vpe_count()

        # Якщо ще немає VPE, використовуємо початковий threshold
        if num_vpe == 0:
            return self.vpe_conf_threshold

        # Лінійна інтерполяція від vpe_conf_threshold до vpe_conf_max
        # Збільшуємо threshold (більший = більш строгий)
        progress = min(1.0, num_vpe / self.vpe_conf_adaptive_rate)
        current_threshold = self.vpe_conf_threshold + (self.vpe_conf_max - self.vpe_conf_threshold) * progress

        return current_threshold

    def _compute_iou(self, bbox1, bbox2):
        """IoU між двома bbox [x1, y1, x2, y2]"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_center_distance(self, bbox1, bbox2):
        """
        Обчислити евклідову відстань між центрами двох bbox [x1, y1, x2, y2]

        Returns:
            float - відстань в пікселях
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Центри bbox
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2

        # Евклідова відстань
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

        return distance

    def _compute_diou(self, bbox1, bbox2):
        """
        Обчислити DIoU (Distance IoU) між двома bbox [x1, y1, x2, y2]

        DIoU = IoU - (d²/c²)
        де:
        - d - відстань між центрами
        - c - діагональ найменшого прямокутника що охоплює обидва bbox

        Returns:
            float - DIoU значення від -1 до 1
        """
        # IoU
        iou = self._compute_iou(bbox1, bbox2)

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Центри bbox
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2

        # Відстань між центрами (d²)
        d_squared = (cx1 - cx2)**2 + (cy1 - cy2)**2

        # Найменший прямокутник що охоплює обидва bbox
        x_min = min(x1_1, x1_2)
        y_min = min(y1_1, y1_2)
        x_max = max(x2_1, x2_2)
        y_max = max(y2_1, y2_2)

        # Діагональ охоплюючого прямокутника (c²)
        c_squared = (x_max - x_min)**2 + (y_max - y_min)**2

        # DIoU
        diou = iou - (d_squared / c_squared if c_squared > 0 else 0)

        return diou

    def reset(self):
        """Скидання стану трекера"""
        super().reset()

        # Reset VPE memory (dual or simple)
        if self.use_dual_memory_vpe:
            self.dual_memory.reset()
        else:
            self.vpe_list.clear()

        self.frame_count = -1
        self.aggregated_vpe = None
        self.last_valid_bbox = None
        self.lost_frames = 0
        self.vpe_pending = False
        self.rejected_candidates = []
        self.search_candidates = []
        self.top_candidates = []
        self.last_bbox_is_kalman_only = False
        self.in_warmup = False
        self.warmup_vpe_collected = 0
        self.kalman = None  # Скинути Калман фільтр
        self.phase1_high_conf_reid_candidate = None
        self.phase1_high_conf_reid_validation_count = 0
        self.phase2_pending_candidate = None
        self.phase2_pending_validation_count = 0
        self.phase3_vpe_freeze_counter = 0
        self.phase3_validation_failure_count = 0
        self.in_phase3_validation = False
        self.phase3_validation_consecutive_successes = 0
        self.phase3_validation_bbox = None
        self.phase3_reference_bbox = None
        self.phase3_kalman_exit_bbox = None

    def get_tracking_info(self) -> Dict[str, Any]:
        """
        Отримати інформацію про стан трекінгу для візуалізації

        Returns:
            Dict з інформацією про поточний стан трекінгу
        """
        info = {
            'lost_frames': self.lost_frames,
            'in_warmup': self.in_warmup,
            'warmup_vpe_collected': self._get_vpe_count() if self.in_warmup else self.warmup_vpe_collected,
        }

        # last_valid_bbox — передаємо завжди коли об'єкт втрачено (Phase 2 і Phase 3)
        if self.lost_frames > 0 and self.last_valid_bbox:
            x1, y1, x2, y2 = self.last_valid_bbox
            info['last_valid_bbox'] = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]

        # Top candidates з IoU scores у Phase 1 для візуалізації
        if self.lost_frames == 0 and len(self.top_candidates) > 0:
            info['top_candidates'] = self.top_candidates

        # Відкинуті кандидати з Фази 3 для візуалізації
        if len(self.rejected_candidates) > 0:
            info['rejected_candidates'] = self.rejected_candidates

        # Всі detections під час Phase 2/3 для візуалізації
        if len(self.search_candidates) > 0:
            info['search_candidates'] = self.search_candidates

        # Флаг: чи є поточний bbox тільки від Калмана (не валідовано детекціями)
        info['last_bbox_is_kalman_only'] = self.last_bbox_is_kalman_only

        # Reference bbox для Phase 3 порівнянь (залежно від phase3_ref_mode)
        if self.phase3_reference_bbox is not None:
            x1, y1, x2, y2 = self.phase3_reference_bbox
            info['phase3_reference_bbox'] = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
            info['phase3_ref_mode'] = self.phase3_ref_mode

        return info

    def _calculate_cosine_similarity(self, vpe1: torch.Tensor, vpe2: torch.Tensor) -> float:
        """
        Розрахувати cosine similarity між двома VPE векторами

        Args:
            vpe1: VPE tensor [1, 1, D] або [1, D]
            vpe2: VPE tensor [1, 1, D] або [1, D]

        Returns:
            Cosine similarity від 0 до 1
        """
        try:
            if vpe1 is None or vpe2 is None:
                return 0.0

            # Перевірити що це тензори, а не tuple/list
            if not isinstance(vpe1, torch.Tensor) or not isinstance(vpe2, torch.Tensor):
                if self.verbose:
                    print(f"   ⚠️  VPE має неправильний тип: vpe1={type(vpe1).__name__}, vpe2={type(vpe2).__name__}")
                return 0.0

            # Flatten до [D]
            v1 = vpe1.flatten()
            v2 = vpe2.flatten()

            # Cosine similarity
            similarity = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            # Масштабувати від [-1, 1] до [0, 1]
            return ((similarity.item() + 1) / 2)
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Помилка при розрахунку cosine similarity: {e}")
            return 0.0

    def _get_all_detections_proximity(self, image: np.ndarray, boxes, tracked_idx: int = None) -> List[Dict]:
        """
        Отримати інформацію про top-7 детекції з proximity до anchor/LT/ST
        Завжди включає tracked об'єкт, навіть якщо він не в top-7 по конфіденції

        Args:
            image: Поточний кадр
            boxes: Детекції з YOLOE
            tracked_idx: Індекс вибраної детекції (той що відстежується)

        Returns:
            Список з інформацією про детекції + proximity (tracked завжди перший)
        """
        if self.lost_frames > 0 or len(boxes) == 0 or not self.use_dual_memory_vpe or self.dual_memory is None:
            return []

        try:
            # Отримати reference VPE
            anchor_vpe = self.dual_memory.get_anchor_vpe()
            lt_avg_vpe = self.dual_memory.get_long_term_avg_vpe()
            st_avg_vpe = self.dual_memory.get_short_term_avg_vpe()

            # Якщо немає ніяких embedding'ів, не розраховувати
            if anchor_vpe is None and lt_avg_vpe is None and st_avg_vpe is None:
                return []

            detections_info = []

            # Top-7 детекцій по конфіденції
            box_list = list(boxes)
            box_list_sorted = sorted(
                enumerate(box_list),
                key=lambda x: float(x[1].conf[0].cpu().numpy()),
                reverse=True
            )[:7]

            # Зібрати індекси з топ-7
            top_indices = set(idx for idx, _ in box_list_sorted)

            # Якщо tracked_idx не в топ-7, додати його
            indices_to_process = box_list_sorted[:]
            if tracked_idx is not None and tracked_idx not in top_indices:
                if tracked_idx < len(box_list):
                    indices_to_process.append((tracked_idx, box_list[tracked_idx]))
                    if self.verbose:
                        print(f"   📌 Tracked об'єкт (idx={tracked_idx}) не в топ-7 по conf, додано окремо")

            for orig_idx, box in indices_to_process:
                try:
                    box_xyxy = box.xyxy[0].cpu().numpy()
                    box_conf = float(box.conf[0].cpu().numpy())

                    # Обчислити IoU з last_valid_bbox
                    iou = self._compute_iou(self.last_valid_bbox, box_xyxy)

                    # Обчислити размір
                    last_valid_w = self.last_valid_bbox[2] - self.last_valid_bbox[0]
                    last_valid_h = self.last_valid_bbox[3] - self.last_valid_bbox[1]
                    box_w = box_xyxy[2] - box_xyxy[0]
                    box_h = box_xyxy[3] - box_xyxy[1]

                    size_ratio_w = box_w / (last_valid_w + 1e-6)
                    size_ratio_h = box_h / (last_valid_h + 1e-6)

                    # Отримати VPE для цієї детекції (окремий predict)
                    anchor_prox = None
                    lt_prox = None
                    st_prox = None

                    try:
                        visual_prompts = dict(
                            bboxes=np.array([box_xyxy]),
                            cls=np.array([0]),
                        )

                        # Використовуємо той самий підхід що і в _collect_vpe
                        current_conf = self._get_adaptive_conf()
                        results = self.model.predict(
                            image,
                            visual_prompts=visual_prompts,
                            predictor=YOLOEVPSegPredictor,
                            conf=current_conf,
                            imgsz=self.imgsz,
                            device=self.device,
                            verbose=False,
                        )

                        # Отримати VPE через predictor (як в _collect_vpe)
                        self.model.predictor.set_prompts(visual_prompts)
                        det_vpe = self.model.predictor.get_vpe(image)

                        if det_vpe is not None:
                            if self.verbose:
                                print(f"      🔍 Det {orig_idx}: VPE shape: {det_vpe.shape}")

                            # Обчислити proximity
                            anchor_prox = self._calculate_cosine_similarity(anchor_vpe, det_vpe) if anchor_vpe is not None else None
                            lt_prox = self._calculate_cosine_similarity(lt_avg_vpe, det_vpe) if lt_avg_vpe is not None else None
                            st_prox = self._calculate_cosine_similarity(st_avg_vpe, det_vpe) if st_avg_vpe is not None else None
                        elif self.verbose:
                            print(f"      ⚠️  Det {orig_idx}: VPE is None")

                    except Exception as e:
                        if self.verbose:
                            print(f"      ⚠️  Det {orig_idx}: помилка VPE - {e}")

                    detection_info = {
                        'idx': orig_idx,
                        'conf': box_conf,
                        'iou': iou,
                        'size_w': size_ratio_w,
                        'size_h': size_ratio_h,
                        'anchor_proximity': anchor_prox,
                        'lt_proximity': lt_prox,
                        'st_proximity': st_prox,
                        'bbox': box_xyxy.tolist(),
                        'vpe': det_vpe,  # Зберегти VPE для подальшого використання
                    }
                    detections_info.append(detection_info)

                except Exception as e:
                    continue

            if self.verbose and len(detections_info) > 0:
                print(f"   📊 Зібрано proximity для {len(detections_info)} детекцій")

            # Відсортувати так, щоб tracked об'єкт був першим
            if tracked_idx is not None:
                detections_info.sort(key=lambda x: (x['idx'] != tracked_idx, -x['conf']))
            else:
                # Якщо tracked_idx не заданий, сортувати по конфіденції
                detections_info.sort(key=lambda x: -x['conf'])

            return detections_info
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Помилка при обчисленні proximity детекцій: {e}")
            return []

    def _get_proximity_info(self) -> Optional[Dict[str, Any]]:
        """
        Отримати інформацію про proximity до anchor та memory для Phase 1

        Returns:
            Dict з proximity даними для debug таблиці або None
        """
        if not self.use_dual_memory_vpe or self.dual_memory is None or self.lost_frames > 0:
            return None

        try:
            anchor_vpe = self.dual_memory.get_anchor_vpe()
            lt_avg_vpe = self.dual_memory.get_long_term_avg_vpe()
            st_avg_vpe = self.dual_memory.get_short_term_avg_vpe()

            # Якщо немає ніяких embedding'ів, не показувати таблицю
            if anchor_vpe is None and lt_avg_vpe is None and st_avg_vpe is None:
                return None

            metadata = self.dual_memory.get_vpe_metadata()

            # Знайти VPE вибраної детекції (той що відстежується зараз)
            current_vpe = None
            if self.selected_detection_idx is not None and len(self.current_frame_detections) > 0:
                # Знайти детекцію з відповідним індексом
                for det in self.current_frame_detections:
                    if det.get('idx') == self.selected_detection_idx:
                        # Отримати збережений VPE для цієї детекції
                        current_vpe = det.get('vpe')
                        if self.verbose and current_vpe is not None:
                            print(f"   📊 Використано VPE вибраної детекції (idx={self.selected_detection_idx}) для proximity")
                        break

            # Fallback до aggregated_vpe якщо не знайдено
            if current_vpe is None:
                current_vpe = self.aggregated_vpe
                if self.verbose:
                    print(f"   ⚠️  VPE вибраної детекції не знайдено, використано aggregated_vpe")

            proximity = {
                'anchor_proximity': self._calculate_cosine_similarity(anchor_vpe, current_vpe) if anchor_vpe is not None else None,
                'lt_proximity': self._calculate_cosine_similarity(lt_avg_vpe, current_vpe) if lt_avg_vpe is not None else None,
                'st_proximity': self._calculate_cosine_similarity(st_avg_vpe, current_vpe) if st_avg_vpe is not None else None,
                'metadata': metadata,
                'detections': self.current_frame_detections,  # Додати proximity для всіх детекцій
                'selected_detection_idx': self.selected_detection_idx,  # Додати індекс вибраної детекції
            }

            return proximity
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Помилка при розрахунку proximity info: {e}")
            return None

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Отримати debug інформацію для візуалізації

        Returns:
            Dict з інформацією про стан трекера та proximity до memory
        """
        # Визначити поточну фазу
        if self.lost_frames == 0:
            phase = "PHASE 1: IoU Matching"
        elif self.lost_frames < self.max_lost_frames:
            phase = f"PHASE 2: Waiting ({self.lost_frames}/{self.max_lost_frames})"
        else:
            phase = "PHASE 3: Re-Identification"

        debug_info = {
            'num_vpe': self._get_vpe_count(),
            'max_vpe': self.max_vpe,
            'vpe_step': self.vpe_step,
            'vpe_conf_threshold': self.vpe_conf_threshold,
            'vpe_pending': self.vpe_pending,
            'frame_count': self.frame_count,
            'iou_threshold': self.iou_threshold,
            'max_lost_frames': self.max_lost_frames,
            'lost_frames': self.lost_frames,
            'current_phase': phase,
            'has_aggregated_vpe': self.aggregated_vpe is not None,
            'use_kalman': self.use_kalman,
            'kalman_active': self.kalman is not None,
        }

        # Додати proximity інформацію для Phase 1 з контролем частоти
        debug_info['debug_frame_step'] = self.debug_frame_step
        debug_info['should_show_by_step'] = (self.frame_count % self.debug_frame_step == 0)

        if self.lost_frames == 0 and self.frame_count % self.debug_frame_step == 0:
            proximity_info = self._get_proximity_info()
            if proximity_info is not None:
                debug_info['proximity_info'] = proximity_info
                debug_info['show_proximity_table'] = True
                if self.verbose:
                    print(f"   📊 Debug table буде показана (frame={self.frame_count}, step={self.debug_frame_step})")
            else:
                debug_info['show_proximity_table'] = False
                if self.verbose:
                    print(f"   ℹ️  proximity_info is None (frame={self.frame_count})")
        else:
            debug_info['show_proximity_table'] = False

        return debug_info


if __name__ == '__main__':
    print("✅ YOLOe-VP-IoU Tracker")
    print(f"   Назва: {YOLOeVPIoUTracker.get_name()}")
    print(f"   Параметри: {YOLOeVPIoUTracker.get_default_params()}")
