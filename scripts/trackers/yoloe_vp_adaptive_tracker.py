"""
YOLOe Visual Prompt Adaptive Tracker - Modular Integration

Трекер для modular_evaluation.py з adaptive VPE collection.

Особливості:
- Збирає VPE з декількох кадрів (з кроком)
- Агрегує їх через усереднення + нормалізацію
- Адаптується до змін appearance
- Підтримує всі параметри через CLI

Використання:
    python scripts/modular_evaluation.py \
        --tracker YOLOe-VP-Adaptive \
        --model yoloe-26s-seg-pf.pt \
        --tracker-params '{"vpe_step": 10, "max_vpe": 5}' \
        --data-dir /path/to/data \
        --output ./results
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List, Optional, Dict, Any, Tuple
import tempfile
import os
from collections import deque

# YOLOe imports
try:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    import torch
    import torch.nn.functional as F
    YOLOE_AVAILABLE = True
except ImportError:
    YOLOE_AVAILABLE = False
    print("⚠️  YOLOE not available. YOLOe-VP-Adaptive tracker will not work.")

from trackers.base_tracker import BaseTracker, register_tracker


@register_tracker
class YOLOeVPAdaptiveTracker(BaseTracker):
    """
    YOLOe Visual Prompt Adaptive Tracker

    Adaptive трекер з динамічним збором та агрегацією VPE.

    ВАЖЛИВО: Використовуйте моделі БЕЗ суфікса '-pf' (prompt-free)!
    Підтримувані моделі: yoloe-26s-seg.pt, yoloe-v8l-seg.pt, etc.
    НЕ підтримуються: yoloe-26s-seg-pf.pt (prompt-free моделі)

    Parameters:
        model_path: str - шлях до YOLOe моделі (БЕЗ '-pf'!)
        conf: float - мінімальна впевненість detection (default: 0.1)
        imgsz: int - розмір вхідного зображення (default: 640)
        device: str - пристрій ('cuda' або 'cpu', default: 'cuda')
        vpe_step: int - крок збору VPE (default: 10)
        max_vpe: int - максимальна кількість VPE (default: 5)
        vpe_conf_threshold: float - мінімальна conf для збору VPE (default: 0.5)
        verbose: bool - виводити деталі (default: False)

    VPE Quality Control:
        VPE збирається тільки якщо detection.conf >= vpe_conf_threshold
        Це забезпечує високу якість агрегованого VPE
    """

    def __init__(self,
                 model_path: str = 'yoloe-26s-seg.pt',
                 conf: float = 0.1,
                 imgsz: int = 640,
                 device: str = 'cuda',
                 vpe_step: int = 10,
                 max_vpe: int = 5,
                 vpe_conf_threshold: float = 0.5,
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
        self.imgsz = imgsz
        self.device = device
        self.vpe_step = vpe_step
        self.max_vpe = max_vpe
        self.vpe_conf_threshold = vpe_conf_threshold
        self.verbose = verbose

        # Ініціалізація моделі
        if self.verbose:
            print(f"📦 Завантаження YOLOe моделі: {model_path}")
        self.model = YOLOE(model_path)
        if hasattr(self.model, 'to'):
            self.model.to(device)

        # VPE collection
        self.vpe_list = deque(maxlen=max_vpe)
        self.frame_count = -1
        self.current_bbox = None
        self.aggregated_vpe = None
        self.initialized = False

        # VPE quality control
        self.vpe_pending = False  # Флаг: чи потрібно зібрати VPE при достатній conf

        if self.verbose:
            print(f"✅ YOLOe-VP-Adaptive готовий (vpe_step={vpe_step}, max_vpe={max_vpe})")

    @classmethod
    def get_name(cls) -> str:
        return "YOLOe-VP-Adaptive"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'model_path': 'yoloe-26s-seg.pt',
            'conf': 0.1,
            'imgsz': 640,
            'device': 'cuda',
            'vpe_step': 10,
            'max_vpe': 5,
            'vpe_conf_threshold': 0.5,
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

            if self.verbose:
                print(f"✅ Ініціалізація з bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

            # Зібрати перший VPE
            self._collect_vpe(image, self.current_bbox)

            self.initialized = True
            self.frame_count = -1

            return True

        except Exception as e:
            print(f"❌ Помилка ініціалізації: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update(self, image: np.ndarray) -> Tuple[bool, Optional[List[float]]]:
        """
        Оновлення на новому кадрі

        Args:
            image: Новий кадр (H, W, 3) BGR

        Returns:
            (success, bbox) - успіх та bbox [x, y, w, h]
        """
        if not self.initialized:
            return False, None

        self.frame_count += 1

        try:
            # Використати агрегований VPE якщо є
            if self.aggregated_vpe is not None:
                # Встановити клас з агрегованим VPE (завжди 0 для VP моделей)
                self.model.is_fused = lambda: False
                self.model.set_classes([0], self.aggregated_vpe)  # Use int instead of string

                # Run prediction
                results = self.model.predict(
                    image,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            else:
                # Fallback: використати звичайні visual prompts
                visual_prompts = dict(
                    bboxes=np.array([self.current_bbox]),
                    cls=np.array([0]),
                )

                results = self.model.predict(
                    image,
                    visual_prompts=visual_prompts,
                    predictor=YOLOEVPSegPredictor,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )

            print(f"len={len(results)}")
            # Знайти найкращий detection (за confidence)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_idx = -1
                best_conf = 0

                for idx, box in enumerate(boxes):
                    box_conf = float(box.conf[0].cpu().numpy())

                    if box_conf > best_conf:
                        best_conf = box_conf
                        best_idx = idx

                if best_idx >= 0:
                    # Знайдено об'єкт
                    best_box = boxes[best_idx]
                    xyxy = best_box.xyxy[0].cpu().numpy()
                    self.current_bbox = xyxy.tolist()

                    # Зібрати VPE якщо час (або pending) та conf достатня
                    should_collect_vpe = (self.frame_count % self.vpe_step == 0) or self.vpe_pending

                    if should_collect_vpe:
                        if best_conf >= self.vpe_conf_threshold:
                            if self.verbose:
                                pending_msg = " (pending)" if self.vpe_pending else ""
                                print(f"🔄 Кадр {self.frame_count}: Збір VPE{pending_msg} (conf={best_conf:.3f} >= {self.vpe_conf_threshold})")
                            self._collect_vpe(image, self.current_bbox)
                            self.vpe_pending = False  # Зібрано, скинути флаг
                        else:
                            self.vpe_pending = True  # Встановити флаг для наступних кадрів
                            if self.verbose:
                                print(f"⚠️  Кадр {self.frame_count}: VPE пропущено (conf={best_conf:.3f} < {self.vpe_conf_threshold}), спроба на наступному кадрі")

                    # Convert to [x, y, w, h]
                    x1, y1, x2, y2 = xyxy
                    return True, [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            # Не знайдено
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
            results = self.model.predict(
                image,
                visual_prompts=visual_prompts,
                predictor=YOLOEVPSegPredictor,
                conf=self.conf,
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

    def _compute_iou(self, bbox1, bbox2):
        """IoU між двома bbox [x1, y1, x2, y2]"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def reset(self):
        """Скидання стану трекера"""
        super().reset()
        self.vpe_list.clear()
        self.frame_count = -1
        self.aggregated_vpe = None
        self.vpe_pending = False

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Отримати debug інформацію для візуалізації

        Returns:
            Dict з інформацією про стан трекера
        """
        return {
            'num_vpe': len(self.vpe_list),
            'max_vpe': self.max_vpe,
            'vpe_step': self.vpe_step,
            'vpe_conf_threshold': self.vpe_conf_threshold,
            'vpe_pending': self.vpe_pending,
            'frame_count': self.frame_count,
            'has_aggregated_vpe': self.aggregated_vpe is not None,
        }
