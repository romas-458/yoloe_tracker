"""
YOLOe Class-based Reinit Tracker - YOLOe з реініціалізацією на основі класів

Особливості:
- Збирає набір класів об'єкта протягом трекінгу
- При втраті об'єкта використовує класи для фільтрування кандидатів
- Може реініціалізуватися на об'єкті з відповідним класом
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List, Optional, Dict, Any, Set, Tuple
import torch
import gc
from collections import Counter

# Додати кастомний ultralytics з YOLOE до path
custom_ultralytics_path = "/home/roman/Projects/basic/ultralytics"
if custom_ultralytics_path not in sys.path:
    sys.path.insert(0, custom_ultralytics_path)

# Додати parent до path для імпорту
sys.path.insert(0, str(Path(__file__).parent.parent))

from trackers.base_tracker import BaseTracker, register_tracker


class KalmanBoxTracker:
    """
    Калман фільтр для bbox трекінгу

    Вектор стану: [x, y, w, h, vx, vy, vw, vh]
    Вимірювання: [x, y, w, h]
    """

    def __init__(self, bbox: List[float], process_noise: float = 0.01, measurement_noise: float = 10.0):
        """
        Ініціалізація Калман фільтра з початковим bbox

        Args:
            bbox: [x, y, w, h]
            process_noise: Шум процесу (вища = швидша адаптація)
            measurement_noise: Шум вимірювання (вища = більше згладжування)
        """
        # Константа для перетворення bbox в центр
        self.kf = cv2.KalmanFilter(8, 4)  # 8 станів, 4 вимірювання

        # Матриця переходу стану (A)
        # x' = x + vx*dt, y' = y + vy*dt, w' = w + vw*dt, h' = h + vh*dt
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1]
        ], dtype=np.float32)

        # Матриця вимірювання (H)
        # Вимірюємо тільки [x, y, w, h]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # Шум процесу (Q) - uncertainty в моделі руху
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise

        # Шум вимірювання (R) - uncertainty в детекції
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise

        # Початкова коваріація помилки (P)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1000.0

        # Ініціалізація стану з bbox [x, y, w, h, 0, 0, 0, 0]
        x, y, w, h = bbox
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> List[float]:
        """
        Прогноз наступного стану

        Returns:
            Прогнозований bbox [x, y, w, h]
        """
        # Прогноз наступного стану
        predicted = self.kf.predict()

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # Повернути bbox [x, y, w, h]
        x, y, w, h = predicted[0:4].flatten()
        return [float(x), float(y), float(w), float(h)]

    def update(self, bbox: List[float]) -> List[float]:
        """
        Оновлення стану з новим вимірюванням

        Args:
            bbox: Виміряний bbox [x, y, w, h]

        Returns:
            Оновлений bbox [x, y, w, h]
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Корекція стану з вимірюванням
        measurement = np.array(bbox, dtype=np.float32).reshape(-1, 1)
        self.kf.correct(measurement)

        # Повернути оновлений bbox
        state = self.kf.statePost
        x, y, w, h = state[0:4].flatten()
        return [float(x), float(y), float(w), float(h)]

    def get_state(self) -> List[float]:
        """
        Отримати поточний стан

        Returns:
            Поточний bbox [x, y, w, h]
        """
        state = self.kf.statePost
        x, y, w, h = state[0:4].flatten()
        return [float(x), float(y), float(w), float(h)]


@register_tracker
class YOLOeClassReinitTracker(BaseTracker):
    """
    YOLOe + IoU matching + реініціалізація на основі класів

    Використовує:
    - YOLOe для генерації bbox кандидатів з класами
    - IoU matching для вибору найкращого
    - Збір набору класів об'єкта
    - Реініціалізація при втраті на основі класів
    """

    def __init__(
        self,
        model_path: str = "yoloe-26s-seg-pf.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        iou_threshold: float = 0.1,
        class_update_interval: int = 5,  # Кожні N кадрів оновлювати класи
        max_lost_frames: int = 10,  # Максимум кадрів без детекції
        min_class_confidence: float = 0.5,  # Мінімальна впевненість класу
        use_class_for_reinit: bool = True,  # Використовувати класи при реініціалізації
        class_filter_mode: str = "all",  # "all" = всі класи з набору, "top3" = топ-3, "dominant" = тільки домінуючий
        reidentify_after_lost: bool = True,  # Спробувати реідентифікацію після max_lost_frames
        reidentify_iou_threshold: float = 0.0,  # Мінімальний IoU для реідентифікації (0.0 = ігнорувати IoU)
        max_reidentify_attempts: int = 30,  # Максимум спроб реідентифікації після втрати
        warmup_frames: int = 10,  # Кількість кадрів для warmup етапу VPE
        warmup_conf: float = 0.15,  # Мінімальний confidence для детекції під час warmup (нижчий за conf)
        vpe_iou_threshold: float = 0.5,  # Мінімальний IoU для додавання VPE
        # Параметри Калман фільтра
        use_kalman: bool = True,  # Використовувати Калман фільтр
        kalman_process_noise: float = 0.01,  # Шум процесу Калмана
        kalman_measurement_noise: float = 10.0,  # Шум вимірювання Калмана
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.class_update_interval = class_update_interval
        self.max_lost_frames = max_lost_frames
        self.min_class_confidence = min_class_confidence
        self.use_class_for_reinit = use_class_for_reinit
        self.class_filter_mode = class_filter_mode
        self.reidentify_after_lost = reidentify_after_lost
        self.reidentify_iou_threshold = reidentify_iou_threshold
        self.max_reidentify_attempts = max_reidentify_attempts
        self.warmup_frames = warmup_frames
        self.warmup_conf = warmup_conf
        self.vpe_iou_threshold = vpe_iou_threshold
        # Параметри Калман фільтра
        self.use_kalman = use_kalman
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.device = device if torch.cuda.is_available() else "cpu"

        # Lazy initialization
        self.model = None
        self._init_model()

        # Трекінг стану
        self.frame_count = -1
        self.lost_frames = 0
        self.reidentify_attempts = 0  # Кількість спроб реідентифікації після втрати
        self.object_classes = Counter()  # Зберігає класи та їх частоту
        self.class_history = []  # Історія класів для аналізу
        self.current_class = None  # Поточний клас об'єкта
        self.current_class_conf = 0.0  # Поточна впевненість класу
        self.in_warmup = False  # Чи знаходимось у warmup періоді
        self.vpe_count = 0  # Лічильник зібраних VPE під час warmup

        # Калман фільтр
        self.kalman = None  # Ініціалізується при инициализации трекера

        # Для візуалізації
        self.previous_bbox = None  # Попередній bbox
        self.top_candidates = []  # Топ-3 кандидати з IoU для візуалізації

    def _init_model(self):
        """Ініціалізація YOLOe моделі"""
        from ultralytics import YOLO

        # YOLO автоматично визначить, що це YOLOE модель за назвою файлу
        self.model = YOLO(self.model_path)

    @classmethod
    def get_name(cls) -> str:
        return "YOLOe-ClassReinit"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            # 'model_path': 'yoloe-11s-seg-pf.pt',

            'model_path': 'yoloe-v8s-seg-pf.pt',
            'imgsz': 640,
            'conf': 0.25,
            'iou_threshold': 0.4,
            'class_update_interval': 5,
            'max_lost_frames': 10,
            'min_class_confidence': 0.5,
            'use_class_for_reinit': True,
            'class_filter_mode': 'all',
            'reidentify_after_lost': True,
            'reidentify_iou_threshold': 0.0,
            'max_reidentify_attempts': 30,
            'warmup_frames': 10,
            'warmup_conf': 0.15,
            'vpe_iou_threshold': 0.5,
            # Параметри Калман фільтра
            'use_kalman': True,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 10.0,
            'device': 'cuda'
        }

    def initialize(self, image: np.ndarray, bbox: List[float]) -> bool:
        """Ініціалізація з першим кадром"""
        self.current_bbox = bbox
        self.initialized = True
        self.frame_count = -1
        self.lost_frames = 0
        self.reidentify_attempts = 0
        self.object_classes = Counter()
        self.class_history = []
        self.in_warmup = True  # Початок warmup періоду
        self.vpe_count = 0  # Скинути лічильник VPE

        # Ініціалізація Калман фільтра
        if self.use_kalman:
            self.kalman = KalmanBoxTracker(
                bbox,
                process_noise=self.kalman_process_noise,
                measurement_noise=self.kalman_measurement_noise
            )

        # Спробувати отримати клас з першого кадру
        candidates = self._detect_bboxes_with_classes(image)
        if candidates:
            # Знайти найближчий bbox до початкового
            best_match = None
            best_iou = 0.0
            for cand_bbox, cand_class, cand_conf in candidates:
                iou = self._compute_iou(bbox, cand_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (cand_bbox, cand_class, cand_conf)

            if best_match and best_iou >= self.iou_threshold:
                _, class_id, class_conf = best_match
                if class_conf >= self.min_class_confidence:
                    self.object_classes[class_id] = 1
                    self.class_history.append(class_id)

        return True

    def update(self, image: np.ndarray) -> Tuple[bool, Optional[List[float]]]:
        """Оновлення на новому кадрі"""
        if not self.initialized:
            return False, None

        self.frame_count += 1

        # Перевірка завершення warmup періоду
        if self.in_warmup and self.frame_count >= self.warmup_frames:
            self.in_warmup = False

        # Зберегти попередній bbox для візуалізації
        self.previous_bbox = self.current_bbox.copy() if self.current_bbox else None

        # 1. Прогноз Калман фільтра (якщо включений)
        reference_bbox = self.current_bbox
        if self.use_kalman and self.kalman:
            reference_bbox = self.kalman.predict()

        # Детекція bbox через YOLOe з класами
        # Під час warmup використовувати нижчий поріг для збору VPE
        if self.in_warmup:
            candidates = self._detect_bboxes_with_classes(image, conf_threshold=self.warmup_conf)
        else:
            candidates = self._detect_bboxes_with_classes(image)

        if not candidates:
            # Немає кандидатів
            self.lost_frames += 1
            self.top_candidates = []  # Очистити кандидатів
            if self.lost_frames <= self.max_lost_frames and self.use_kalman and self.kalman:
                # Використовувати прогноз Калман фільтра
                return False, reference_bbox
            elif self.lost_frames <= self.max_lost_frames:
                return False, self.current_bbox
            else:
                # Об'єкт втрачено
                return False, None

        # Обчислити IoU для всіх кандидатів з попереднім bbox для візуалізації
        # (використовувати Калман прогноз, якщо доступний)
        self._compute_top_candidates(reference_bbox, candidates)

        # IoU matching з використанням Калман прогнозу як reference
        best_bbox, best_iou, best_class, best_conf = self._find_best_match(
            reference_bbox,
            candidates,
            use_class_filter=False  # Спочатку без фільтру класів
        )

        # Позначити який candidate був вибраний як best у top_candidates
        if best_bbox:
            for cand in self.top_candidates:
                if (abs(cand['bbox'][0] - best_bbox[0]) < 0.1 and
                    abs(cand['bbox'][1] - best_bbox[1]) < 0.1):
                    cand['is_best_match'] = True
                else:
                    cand['is_best_match'] = False

        if best_iou >= self.iou_threshold:
            # Успішний match
            # Оновити Калман фільтр (якщо включений)
            if self.use_kalman and self.kalman:
                self.current_bbox = self.kalman.update(best_bbox)
            else:
                self.current_bbox = best_bbox

            self.lost_frames = 0

            # Зберегти поточний клас та впевненість для візуалізації
            self.current_class = best_class
            self.current_class_conf = best_conf

            # Оновлення класів
            if self.in_warmup:
                # Під час warmup: завжди збирати класи (кожен кадр)
                if best_conf >= self.min_class_confidence:
                    self.object_classes[best_class] += 1
                    self.class_history.append(best_class)

                # Збір VPE під час warmup
                vpe_collected = self._collect_vpe_during_warmup(self.previous_bbox, candidates)
                self.vpe_count += vpe_collected
            else:
                # Після warmup: звичайний режим (кожні N кадрів)
                if self.frame_count % self.class_update_interval == 0:
                    if best_conf >= self.min_class_confidence:
                        self.object_classes[best_class] += 1
                        self.class_history.append(best_class)

            return True, self.current_bbox

        else:
            # Низький IoU - спробувати реініціалізацію на основі класів
            self.lost_frames += 1

            if self.use_class_for_reinit and self.object_classes and self.lost_frames <= self.max_lost_frames:
                # Спробувати знайти об'єкт за класом з послабленим IoU порогом
                reinit_bbox, reinit_iou, reinit_class, reinit_conf = self._find_best_match(
                    self.current_bbox,
                    candidates,
                    use_class_filter=True
                )

                if reinit_bbox and reinit_iou >= self.iou_threshold * 0.5:  # Послабленний поріг
                    # Реініціалізація успішна
                    self.current_bbox = reinit_bbox
                    self.lost_frames = 0
                    self.reidentify_attempts = 0  # Скинути лічильник

                    # Зберегти поточний клас та впевненість
                    self.current_class = reinit_class
                    self.current_class_conf = reinit_conf

                    # Позначити який candidate був вибраний як best у top_candidates
                    for cand in self.top_candidates:
                        if (abs(cand['bbox'][0] - reinit_bbox[0]) < 0.1 and
                            abs(cand['bbox'][1] - reinit_bbox[1]) < 0.1):
                            cand['is_best_match'] = True
                        else:
                            cand['is_best_match'] = False

                    # Оновити класи
                    if reinit_conf >= self.min_class_confidence:
                        self.object_classes[reinit_class] += 1
                        self.class_history.append(reinit_class)

                    return True, reinit_bbox

            # Не вдалося реініціалізувати
            if self.lost_frames <= self.max_lost_frames:
                return False, self.current_bbox
            else:
                # lost_frames > max_lost_frames
                # Спробувати реідентифікацію тільки по класах (без IoU порогу)
                if (self.reidentify_after_lost and self.object_classes and
                    self.reidentify_attempts < self.max_reidentify_attempts):

                    self.reidentify_attempts += 1

                    # Шукати тільки серед кандидатів з відповідними класами
                    reinit_bbox, reinit_iou, reinit_class, reinit_conf = self._find_best_match(
                        self.current_bbox,
                        candidates,
                        use_class_filter=True
                    )

                    # Перевірити тільки мінімальний IoU поріг для реідентифікації
                    if reinit_bbox and reinit_iou >= self.reidentify_iou_threshold:
                        # Реідентифікація успішна!
                        self.current_bbox = reinit_bbox
                        self.lost_frames = 0
                        self.reidentify_attempts = 0  # Скинути лічильник

                        # Зберегти поточний клас та впевненість
                        self.current_class = reinit_class
                        self.current_class_conf = reinit_conf

                        # Позначити який candidate був вибраний як best у top_candidates
                        for cand in self.top_candidates:
                            if (abs(cand['bbox'][0] - reinit_bbox[0]) < 0.1 and
                                abs(cand['bbox'][1] - reinit_bbox[1]) < 0.1):
                                cand['is_best_match'] = True
                            else:
                                cand['is_best_match'] = False

                        # Оновити класи
                        if reinit_conf >= self.min_class_confidence:
                            self.object_classes[reinit_class] += 1
                            self.class_history.append(reinit_class)

                        return True, reinit_bbox

                # Не вдалося реідентифікувати або досягнуто ліміту спроб
                return False, None

    def _detect_bboxes_with_classes(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Tuple[List[float], int, float]]:
        """
        Детекція bbox через YOLOe з класами

        Args:
            image: Вхідне зображення
            conf_threshold: Опціональний поріг confidence (якщо None, використовується self.conf)

        Returns:
            List of (bbox, class_id, confidence)
        """
        detections = []

        # Використати custom conf або стандартний
        conf = conf_threshold if conf_threshold is not None else self.conf

        try:
            with torch.no_grad():
                results = self.model.predict(
                    image,
                    imgsz=self.imgsz,
                    conf=conf,
                    verbose=False,
                    device=self.device
                )

            if results:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else None
                        confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else None

                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                            class_id = int(classes[i]) if classes is not None else -1
                            conf = float(confidences[i]) if confidences is not None else 1.0

                            detections.append((bbox, class_id, conf))

        except Exception as e:
            print(f"Detection error: {e}")

        # Очищення
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return detections

    def _find_best_match(
        self,
        reference: List[float],
        candidates: List[Tuple[List[float], int, float]],
        use_class_filter: bool = False
    ) -> Tuple[Optional[List[float]], float, Optional[int], Optional[float]]:
        """
        Знайти найкращий bbox за IoU (опціонально з фільтром класів)

        Returns:
            (best_bbox, best_iou, best_class, best_conf)
        """
        best_bbox = None
        best_iou = 0.0
        best_class = None
        best_conf = None

        # Фільтрування за класами якщо потрібно
        if use_class_filter and self.object_classes:
            allowed_classes = self._get_allowed_classes()

            filtered_candidates = [
                (bbox, cls, conf) for bbox, cls, conf in candidates
                if cls in allowed_classes
            ]

            # Якщо після фільтрування нічого не залишилось, використовувати всі
            if filtered_candidates:
                candidates = filtered_candidates

        for candidate_bbox, candidate_class, candidate_conf in candidates:
            iou = self._compute_iou(reference, candidate_bbox)
            if iou > best_iou:
                best_iou = iou
                best_bbox = candidate_bbox
                best_class = candidate_class
                best_conf = candidate_conf

        return best_bbox, best_iou, best_class, best_conf

    def _get_allowed_classes(self) -> Set[int]:
        """
        Отримати набір дозволених класів на основі class_filter_mode

        Returns:
            Set з ID класів
        """
        if not self.object_classes:
            return set()

        if self.class_filter_mode == "dominant":
            # Тільки домінуючий клас
            dominant = self.object_classes.most_common(1)[0][0]
            return {dominant}
        elif self.class_filter_mode == "top3":
            # Топ-3 найчастіші класи
            return set([cls for cls, _ in self.object_classes.most_common(3)])
        else:  # "all"
            # Всі класи з набору
            return set(self.object_classes.keys())

    def _collect_vpe_during_warmup(
        self,
        reference: List[float],
        candidates: List[Tuple[List[float], int, float]]
    ) -> int:
        """
        Збір Virtual Positive Examples під час warmup періоду

        Додає до object_classes детекції з:
        - IoU >= vpe_iou_threshold (високий IoU з попереднім bbox)
        - confidence < conf (нижча впевненість, VPE)

        Це детекції які мають високе перекриття з попереднім bbox,
        але низьку впевненість, тому не пройшли б звичайну детекцію.

        Args:
            reference: Попередній bbox [x, y, w, h]
            candidates: Список (bbox, class_id, confidence)

        Returns:
            Кількість зібраних VPE
        """
        vpe_collected = 0

        for bbox, class_id, confidence in candidates:
            # Перевірка умов для VPE:
            # 1. IoU з попереднім bbox >= порогу
            # 2. Впевненість < основного порогу (це VPE, не основна детекція)
            iou = self._compute_iou(reference, bbox)

            if iou >= self.vpe_iou_threshold and confidence < self.conf:
                # Це VPE - додати до класів
                if confidence >= self.min_class_confidence:
                    self.object_classes[class_id] += 1
                    self.class_history.append(class_id)
                    vpe_collected += 1

        return vpe_collected

    def _compute_top_candidates(
        self,
        reference: Optional[List[float]],
        candidates: List[Tuple[List[float], int, float]]
    ):
        """
        Обчислити топ-3 кандидати з найбільшим IoU для візуалізації

        Args:
            reference: Попередній bbox [x, y, w, h]
            candidates: Список (bbox, class_id, confidence)
        """
        if not reference or not candidates:
            self.top_candidates = []
            return

        # Обчислити IoU для всіх кандидатів
        candidates_with_iou = []
        for bbox, class_id, conf in candidates:
            iou = self._compute_iou(reference, bbox)
            candidates_with_iou.append({
                'bbox': bbox,
                'class_id': class_id,
                'confidence': conf,
                'iou': iou
            })

        # Сортувати за IoU та взяти топ-3
        candidates_with_iou.sort(key=lambda x: x['iou'], reverse=True)
        self.top_candidates = candidates_with_iou[:3]


    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """IoU між двома bbox [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        area1, area2 = w1 * h1, w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def get_object_classes(self) -> Dict[int, int]:
        """Отримати набір класів об'єкта з частотами"""
        return dict(self.object_classes)

    def get_dominant_class(self) -> Optional[int]:
        """Отримати домінантний клас об'єкта"""
        if not self.object_classes:
            return None
        return self.object_classes.most_common(1)[0][0]

    def get_tracking_info(self) -> Dict[str, Any]:
        """
        Отримати інформацію про поточний стан трекінгу для візуалізації

        Returns:
            Dict з інформацією: current_class, class_conf, all_classes, previous_bbox, candidates, etc.
        """
        return {
            'current_class': self.current_class,
            'current_class_conf': self.current_class_conf,
            'object_classes': dict(self.object_classes),
            'dominant_class': self.get_dominant_class(),
            'allowed_classes': list(self._get_allowed_classes()) if self.object_classes else [],
            'class_filter_mode': self.class_filter_mode,
            'lost_frames': self.lost_frames,
            'reidentify_attempts': self.reidentify_attempts,
            'frame_count': self.frame_count,
            'previous_bbox': self.previous_bbox,
            'top_candidates': self.top_candidates,  # Список dict з bbox, class_id, conf, iou
            'in_warmup': self.in_warmup,  # Чи в warmup періоді
            'vpe_count': self.vpe_count,  # Кількість зібраних VPE
            'use_kalman': self.use_kalman,  # Чи використовується Калман фільтр
            'kalman_age': self.kalman.age if self.kalman else 0  # Вік Калман фільтра
        }

    def reset(self):
        """Скидання стану"""
        super().reset()
        self.frame_count = -1
        self.lost_frames = 0
        self.reidentify_attempts = 0
        self.object_classes = Counter()
        self.class_history = []
        self.current_class = None
        self.current_class_conf = 0.0
        self.previous_bbox = None
        self.top_candidates = []
        self.in_warmup = False  # Скинути warmup стан
        self.vpe_count = 0  # Скинути лічильник VPE
        self.kalman = None  # Скинути Калман фільтр
        # Model залишається ініціалізованим для швидкості
