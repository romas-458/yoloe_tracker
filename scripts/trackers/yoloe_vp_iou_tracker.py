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
                 phase2_diou_threshold: float = 1.0,
                 waiting_reinit_conf_threshold: float = 1.0,
                 waiting_reinit_diou_threshold: float = -0.5,
                 reinit_diou_threshold: float = -1.0,
                 reinit_diou_max: float = -0.9,
                 reinit_adaptive_rate: int = 15,
                 reinit_conf_threshold: float = 1.0,
                 verbose: bool = False,
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
        self.phase2_diou_threshold = phase2_diou_threshold
        self.waiting_reinit_conf_threshold = waiting_reinit_conf_threshold
        self.waiting_reinit_diou_threshold = waiting_reinit_diou_threshold
        self.reinit_diou_threshold = reinit_diou_threshold
        self.reinit_diou_max = reinit_diou_max
        self.reinit_adaptive_rate = reinit_adaptive_rate
        self.reinit_conf_threshold = reinit_conf_threshold
        self.verbose = verbose

        # Ініціалізація моделі
        if self.verbose:
            print(f"📦 Завантаження YOLOe моделі: {model_path}")
        self.model = YOLOE(model_path)
        if hasattr(self.model, 'to'):
            self.model.to(device)

        # VPE collection
        self.vpe_list = deque(maxlen=max_vpe)
        self.frame_count = 0
        self.current_bbox = None
        self.aggregated_vpe = None
        self.initialized = False

        # Трифазна логіка tracking
        self.last_valid_bbox = None  # Останній валідний bbox (для IoU порівняння)
        self.lost_frames = 0  # Лічильник кадрів без успішного matching

        # VPE quality control
        self.vpe_pending = False  # Флаг: чи потрібно зібрати VPE при достатній conf

        # Debug info для візуалізації
        self.rejected_candidates = []  # Відкинуті кандидати у Фазі 3
        self.search_candidates = []  # Всі detections під час Phase 2/3 для візуалізації

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

            print(f"✅ YOLOe-VP-IoU готовий (vpe_step={vpe_step}, max_vpe={max_vpe}, iou_threshold={iou_threshold}, max_lost_frames={max_lost_frames}{conf_msg}{vpe_conf_msg}{phase2_msg}{waiting_reinit_msg}{reinit_msg}{reinit_conf_msg})")

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
            'phase2_diou_threshold': 1.0,
            'waiting_reinit_conf_threshold': 1.0,
            'waiting_reinit_diou_threshold': -0.5,
            'reinit_diou_threshold': -1.0,
            'reinit_diou_max': -0.9,
            'reinit_adaptive_rate': 15,
            'verbose': False,
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
            self._collect_vpe(image, self.current_bbox)

            self.initialized = True
            self.frame_count = 0

            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Помилка ініціалізації: {e}")
            return False

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

        self.frame_count += 1

        try:
            # ========================================
            # VP Detection (знаходження кандидатів)
            # ========================================
            if self.aggregated_vpe is not None:
                # Використати агрегований VPE
                self.model.is_fused = lambda: False
                self.model.set_classes(["0"], self.aggregated_vpe)

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

            if len(results) == 0 or len(results[0].boxes) == 0:
                # Немає detections
                self.lost_frames += 1

                if self.verbose:
                    print(f"⚠️  Кадр {self.frame_count}: No detections (lost={self.lost_frames}/{self.max_lost_frames})")

                if self.lost_frames >= self.max_lost_frames:
                    # Занадто довго без detections - fail
                    if self.verbose:
                        print(f"❌ Кадр {self.frame_count}: Tracking lost (no detections)")
                    return False, None
                else:
                    # Повернути last_valid_bbox (очікування)
                    x1, y1, x2, y2 = self.last_valid_bbox
                    return True, [x1, y1, x2 - x1, y2 - y1]

            boxes = results[0].boxes

            # ========================================
            # Фаза 1: IoU Matching з last_valid_bbox
            # ========================================
            best_iou_idx = -1
            best_iou = 0
            best_conf_idx = -1
            best_conf = 0

            for idx, box in enumerate(boxes):
                box_xyxy = box.xyxy[0].cpu().numpy()
                box_conf = float(box.conf[0].cpu().numpy())

                # Обчислити IoU з last_valid_bbox (не з current!)
                iou = self._compute_iou(self.last_valid_bbox, box_xyxy)

                if iou > best_iou:
                    best_iou = iou
                    best_iou_idx = idx

                if box_conf > best_conf:
                    best_conf = box_conf
                    best_conf_idx = idx

            # Перевірка IoU threshold
            if best_iou >= self.iou_threshold:
                # ========================================
                # ФАЗА 1: УСПІШНИЙ IoU MATCHING
                # ========================================
                best_box = boxes[best_iou_idx]
                box_xyxy = best_box.xyxy[0].cpu().numpy()
                box_conf = float(best_box.conf[0].cpu().numpy())

                self.current_bbox = box_xyxy.tolist()
                self.last_valid_bbox = box_xyxy.tolist()  # Оновити валідний bbox
                self.lost_frames = 0  # Reset counter
                self.search_candidates = []  # Очистити кандидатів (Phase 1 - знайдено)

                if self.verbose:
                    print(f"✅ Кадр {self.frame_count}: [PHASE 1] IoU Match (IoU={best_iou:.3f}, Conf={box_conf:.3f})")

                # Зібрати VPE якщо час (або pending) та conf достатня
                should_collect_vpe = (self.frame_count % self.vpe_step == 0) or self.vpe_pending

                if should_collect_vpe:
                    current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()
                    if box_conf >= current_vpe_threshold:
                        if self.verbose:
                            pending_msg = " (pending)" if self.vpe_pending else ""
                            print(f"🔄 Кадр {self.frame_count}: Збір VPE{pending_msg} (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe})")
                        self._collect_vpe(image, self.current_bbox)
                        self.vpe_pending = False  # Зібрано, скинути флаг
                    else:
                        self.vpe_pending = True  # Встановити флаг для наступних кадрів
                        if self.verbose:
                            print(f"⚠️  Кадр {self.frame_count}: VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe}), спроба на наступному кадрі")

                x1, y1, x2, y2 = self.current_bbox
                return True, [x1, y1, x2 - x1, y2 - y1]

            else:
                # IoU < threshold - об'єкт не знайдено за geometric matching
                self.lost_frames += 1

                if self.lost_frames < self.max_lost_frames:
                    # ========================================
                    # ФАЗА 2: ПОШУК-ОЧІКУВАННЯ з DIoU
                    # ========================================

                    # Зберегти всі detections для візуалізації
                    self.search_candidates = []
                    for box in boxes:
                        box_xyxy = box.xyxy[0].cpu().numpy()
                        box_conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = box_xyxy
                        diou = self._compute_diou(self.last_valid_bbox, box_xyxy) if self.last_valid_bbox else -float('inf')
                        self.search_candidates.append({
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'conf': box_conf,
                            'diou': diou,
                            'phase': 2
                        })

                    # Спробувати DIoU matching якщо увімкнено
                    phase2_diou_idx = -1
                    phase2_diou_value = -float('inf')

                    if self.phase2_diou_threshold < 1.0 and self.last_valid_bbox:
                        # DIoU увімкнено для Phase 2 (threshold < 1.0)
                        for idx, box in enumerate(boxes):
                            box_xyxy = box.xyxy[0].cpu().numpy()
                            diou = self._compute_diou(self.last_valid_bbox, box_xyxy)

                            if diou >= self.phase2_diou_threshold and diou > phase2_diou_value:
                                phase2_diou_value = diou
                                phase2_diou_idx = idx

                    # Якщо знайдено детекцію за DIoU
                    if phase2_diou_idx >= 0:
                        best_box = boxes[phase2_diou_idx]
                        box_xyxy = best_box.xyxy[0].cpu().numpy()
                        box_conf = float(best_box.conf[0].cpu().numpy())

                        self.current_bbox = box_xyxy.tolist()
                        self.last_valid_bbox = box_xyxy.tolist()
                        self.lost_frames = 0  # Reset counter
                        # Не очищаємо search_candidates тут - візуалізація покаже що було знайдено серед кандидатів

                        if self.verbose:
                            print(f"✅ Кадр {self.frame_count}: [PHASE 2] DIoU Match (IoU={best_iou:.3f}, DIoU={phase2_diou_value:.3f} >= {self.phase2_diou_threshold}, Conf={box_conf:.3f})")

                        # Зібрати VPE якщо час (або pending) та conf достатня
                        should_collect_vpe = (self.frame_count % self.vpe_step == 0) or self.vpe_pending

                        if should_collect_vpe:
                            current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()
                            if box_conf >= current_vpe_threshold:
                                if self.verbose:
                                    pending_msg = " (pending)" if self.vpe_pending else ""
                                    print(f"🔄 Кадр {self.frame_count}: Збір VPE{pending_msg} (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe})")
                                self._collect_vpe(image, self.current_bbox)
                                self.vpe_pending = False
                            else:
                                self.vpe_pending = True
                                if self.verbose:
                                    print(f"⚠️  Кадр {self.frame_count}: VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe}), спроба на наступному кадрі")

                        x1, y1, x2, y2 = self.current_bbox
                        return True, [x1, y1, x2 - x1, y2 - y1]

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

                        self.current_bbox = box_xyxy.tolist()
                        self.last_valid_bbox = box_xyxy.tolist()
                        self.lost_frames = 0  # Reset counter

                        if self.verbose:
                            print(f"🔄 Кадр {self.frame_count}: [PHASE 2] Early Reinit (conf={box_conf:.3f} >= {self.waiting_reinit_conf_threshold}, DIoU={waiting_reinit_diou:.3f} >= {self.waiting_reinit_diou_threshold})")

                        # Зібрати VPE для нового bbox (якщо conf достатня)
                        current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()
                        if box_conf >= current_vpe_threshold:
                            if self.verbose:
                                print(f"   📥 Збір VPE для нового bbox (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe})")
                            self._collect_vpe(image, self.current_bbox)
                            self.vpe_pending = False
                        else:
                            self.vpe_pending = True
                            if self.verbose:
                                print(f"   ⚠️  VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe}), спроба на наступному кадрі")

                        x1, y1, x2, y2 = self.current_bbox
                        return True, [x1, y1, x2 - x1, y2 - y1]

                    # DIoU не спрацював або вимкнено - режим очікування
                    if self.verbose:
                        diou_msg = f", DIoU={phase2_diou_value:.3f} < {self.phase2_diou_threshold}" if self.phase2_diou_threshold < 1.0 else ""
                        print(f"⚠️  Кадр {self.frame_count}: [PHASE 2] Waiting (IoU={best_iou:.3f} < {self.iou_threshold}{diou_msg}, lost={self.lost_frames}/{self.max_lost_frames})")

                    # Не оновлюємо bbox, повертаємо False (втрачений для евалюації)
                    # last_valid_bbox залишається для можливої реідентифікації
                    return False, None

                else:
                    # ========================================
                    # ФАЗА 3: РЕІДЕНТИФІКАЦІЯ
                    # ========================================

                    # Очистити попередні відкинуті кандидати
                    self.rejected_candidates = []

                    # Зберегти всі detections для візуалізації
                    self.search_candidates = []
                    for box in boxes:
                        box_xyxy = box.xyxy[0].cpu().numpy()
                        box_conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = box_xyxy
                        diou = self._compute_diou(self.last_valid_bbox, box_xyxy) if self.last_valid_bbox else -float('inf')
                        self.search_candidates.append({
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'conf': box_conf,
                            'diou': diou,
                            'phase': 3
                        })

                    # Спочатку перевіряємо чи є детекція з високою conf (автоматична реініціалізація)
                    high_conf_candidate_idx = -1
                    high_conf_candidate_conf = 0

                    if self.reinit_conf_threshold < 1.0:
                        for idx, box in enumerate(boxes):
                            box_conf = float(box.conf[0].cpu().numpy())
                            if box_conf >= self.reinit_conf_threshold:
                                if box_conf > high_conf_candidate_conf:
                                    high_conf_candidate_conf = box_conf
                                    high_conf_candidate_idx = idx

                    # Якщо знайдено високоякісну детекцію, використати її
                    if high_conf_candidate_idx >= 0:
                        best_box = boxes[high_conf_candidate_idx]
                        box_xyxy = best_box.xyxy[0].cpu().numpy()
                        box_conf = float(best_box.conf[0].cpu().numpy())

                        self.current_bbox = box_xyxy.tolist()
                        self.last_valid_bbox = box_xyxy.tolist()
                        self.lost_frames = 0  # Reset counter

                        if self.verbose:
                            print(f"🔄 Кадр {self.frame_count}: [PHASE 3] High-Conf Re-ID (conf={box_conf:.3f} >= {self.reinit_conf_threshold})")

                        # Зібрати VPE для нового bbox (якщо conf достатня)
                        current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()
                        if box_conf >= current_vpe_threshold:
                            if self.verbose:
                                print(f"   📥 Збір VPE для нового bbox (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe})")
                            self._collect_vpe(image, self.current_bbox)
                            self.vpe_pending = False  # VPE зібрано
                        else:
                            self.vpe_pending = True  # Встановити флаг для наступних кадрів
                            if self.verbose:
                                print(f"   ⚠️  VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe}), спроба на наступному кадрі")

                        x1, y1, x2, y2 = self.current_bbox
                        return True, [x1, y1, x2 - x1, y2 - y1]

                    # Якщо встановлено reinit_diou_threshold, фільтруємо кандидатів за DIoU
                    reinit_candidate_idx = -1
                    reinit_candidate_conf = 0
                    reinit_candidate_diou = -float('inf')

                    if self.reinit_diou_threshold > -1.0 and self.last_valid_bbox:
                        # Обчислити адаптивний threshold на основі lost_frames
                        current_threshold = self._get_adaptive_diou_threshold()

                        # Фільтрація за DIoU від last_valid_bbox
                        all_candidates = []  # Всі кандидати для verbose виводу

                        for idx, box in enumerate(boxes):
                            box_xyxy = box.xyxy[0].cpu().numpy()
                            box_conf = float(box.conf[0].cpu().numpy())

                            # Обчислити DIoU з last_valid_bbox
                            diou = self._compute_diou(self.last_valid_bbox, box_xyxy)

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
                                if box_conf > reinit_candidate_conf:
                                    reinit_candidate_conf = box_conf
                                    reinit_candidate_idx = idx
                                    reinit_candidate_diou = diou
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

                            print(f"   🔍 Фаза 3: {len(all_candidates)} кандидатів (threshold {threshold_info}):")
                            for i, cand in enumerate(all_candidates):
                                status = "✅ ПРИЙНЯТО" if cand['accepted'] else "❌ ВІДКИНУТО"
                                print(f"      #{i+1}: {status} | conf={cand['conf']:.3f}, DIoU={cand['diou']:.3f}")

                        if reinit_candidate_idx >= 0:
                            # Знайдено кандидата з достатнім DIoU
                            best_box = boxes[reinit_candidate_idx]
                            box_xyxy = best_box.xyxy[0].cpu().numpy()
                            box_conf = float(best_box.conf[0].cpu().numpy())

                            self.current_bbox = box_xyxy.tolist()
                            self.last_valid_bbox = box_xyxy.tolist()  # Новий валідний bbox
                            self.lost_frames = 0  # Reset counter

                            if self.verbose:
                                print(f"🔄 Кадр {self.frame_count}: [PHASE 3] Re-ID (conf={box_conf:.3f}, DIoU={reinit_candidate_diou:.3f} >= {current_threshold:.3f})")

                            # Зібрати VPE для нового bbox (якщо conf достатня)
                            current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()
                            if box_conf >= current_vpe_threshold:
                                if self.verbose:
                                    print(f"   📥 Збір VPE для нового bbox (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe})")
                                self._collect_vpe(image, self.current_bbox)
                                self.vpe_pending = False  # VPE зібрано
                            else:
                                self.vpe_pending = True  # Встановити флаг для наступних кадрів
                                if self.verbose:
                                    print(f"   ⚠️  VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe}), спроба на наступному кадрі")

                            x1, y1, x2, y2 = self.current_bbox
                            return True, [x1, y1, x2 - x1, y2 - y1]
                        else:
                            # Всі detections з низьким DIoU
                            if self.verbose:
                                print(f"❌ Кадр {self.frame_count}: [PHASE 3] Re-ID failed (всі кандидати: DIoU < {current_threshold:.3f})")
                            return False, None

                    else:
                        # Немає обмеження на DIoU - використати max(conf) як раніше
                        if best_conf_idx >= 0:
                            best_box = boxes[best_conf_idx]
                            box_xyxy = best_box.xyxy[0].cpu().numpy()
                            box_conf = float(best_box.conf[0].cpu().numpy())

                            self.current_bbox = box_xyxy.tolist()
                            self.last_valid_bbox = box_xyxy.tolist()  # Новий валідний bbox
                            self.lost_frames = 0  # Reset counter

                            if self.verbose:
                                print(f"🔄 Кадр {self.frame_count}: [PHASE 3] Re-ID (max_conf={box_conf:.3f})")

                            # Зібрати VPE для нового bbox (якщо conf достатня)
                            current_vpe_threshold = self._get_adaptive_vpe_conf_threshold()
                            if box_conf >= current_vpe_threshold:
                                if self.verbose:
                                    print(f"   📥 Збір VPE для нового bbox (conf={box_conf:.3f} >= {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe})")
                                self._collect_vpe(image, self.current_bbox)
                                self.vpe_pending = False  # VPE зібрано
                            else:
                                self.vpe_pending = True  # Встановити флаг для наступних кадрів
                                if self.verbose:
                                    print(f"   ⚠️  VPE пропущено (conf={box_conf:.3f} < {current_vpe_threshold:.3f}, VPE={len(self.vpe_list)}/{self.max_vpe}), спроба на наступному кадрі")

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

    def _collect_vpe(self, image: np.ndarray, bbox: list):
        """
        Зібрати VPE з поточного кадру

        Args:
            image: Поточний кадр
            bbox: Bbox [x1, y1, x2, y2]
        """
        try:
            # Створити visual prompts (завжди використовуємо cls=0 для VP моделей)
            visual_prompts = dict(
                bboxes=np.array([bbox]),
                cls=np.array([0]),
            )

            # Run prediction для створення predictor
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

            # Встановити промти та отримати VPE
            self.model.predictor.set_prompts(visual_prompts)
            vpe = self.model.predictor.get_vpe(image)

            # Додати до списку
            self.vpe_list.append(vpe)

            if self.verbose:
                print(f"   📥 Зібрано VPE #{len(self.vpe_list)}")

            # Агрегувати VPE
            self._aggregate_vpe()

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Помилка збору VPE: {e}")

    def _aggregate_vpe(self):
        """
        Агрегувати всі зібрані VPE
        """
        if len(self.vpe_list) == 0:
            return

        try:
            # Об'єднати та усереднити
            vpe_tensor = torch.cat(list(self.vpe_list), dim=0)
            self.aggregated_vpe = vpe_tensor.mean(dim=0, keepdim=True)

            # Нормалізувати
            self.aggregated_vpe = F.normalize(self.aggregated_vpe, p=2, dim=-1)

            if self.verbose:
                print(f"   🔄 Агреговано {len(self.vpe_list)} VPE")

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Помилка агрегації: {e}")

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
        num_vpe = len(self.vpe_list)

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
        num_vpe = len(self.vpe_list)

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
        self.vpe_list.clear()
        self.frame_count = 0
        self.aggregated_vpe = None
        self.last_valid_bbox = None
        self.lost_frames = 0
        self.vpe_pending = False
        self.rejected_candidates = []
        self.search_candidates = []

    def get_tracking_info(self) -> Dict[str, Any]:
        """
        Отримати інформацію про стан трекінгу для візуалізації

        Returns:
            Dict з інформацією про поточний стан трекінгу
        """
        info = {
            'lost_frames': self.lost_frames,
        }

        # У Фазі 2 (пошук/очікування) передаємо last_valid_bbox для візуалізації
        if self.lost_frames > 0 and self.lost_frames < self.max_lost_frames:
            if self.last_valid_bbox:
                x1, y1, x2, y2 = self.last_valid_bbox
                info['last_valid_bbox'] = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]

        # Відкинуті кандидати з Фази 3 для візуалізації
        if len(self.rejected_candidates) > 0:
            info['rejected_candidates'] = self.rejected_candidates

        # Всі detections під час Phase 2/3 для візуалізації
        if len(self.search_candidates) > 0:
            info['search_candidates'] = self.search_candidates

        return info

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Отримати debug інформацію для візуалізації

        Returns:
            Dict з інформацією про стан трекера
        """
        # Визначити поточну фазу
        if self.lost_frames == 0:
            phase = "PHASE 1: IoU Matching"
        elif self.lost_frames < self.max_lost_frames:
            phase = f"PHASE 2: Waiting ({self.lost_frames}/{self.max_lost_frames})"
        else:
            phase = "PHASE 3: Re-Identification"

        return {
            'num_vpe': len(self.vpe_list),
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
        }


if __name__ == '__main__':
    print("✅ YOLOe-VP-IoU Tracker")
    print(f"   Назва: {YOLOeVPIoUTracker.get_name()}")
    print(f"   Параметри: {YOLOeVPIoUTracker.get_default_params()}")
