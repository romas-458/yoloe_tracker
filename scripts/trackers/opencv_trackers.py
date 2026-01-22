"""
OpenCV Trackers - wrappers для вбудованих трекерів OpenCV

Підтримувані:
- KCF (Kernelized Correlation Filters)
- CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
- MedianFlow
- MOSSE (Minimum Output Sum of Squared Error)
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any

from trackers.base_tracker import BaseTracker, register_tracker


class OpenCVTrackerBase(BaseTracker):
    """
    Базовий клас для OpenCV трекерів
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cv_tracker = None

    def initialize(self, image: np.ndarray, bbox: List[float]) -> bool:
        """Ініціалізація OpenCV трекера"""
        if self.cv_tracker is None:
            self.cv_tracker = self._create_tracker()

        # OpenCV використовує (x, y, w, h) як tuple
        x, y, w, h = bbox
        success = self.cv_tracker.init(image, (int(x), int(y), int(w), int(h)))

        if success:
            self.current_bbox = bbox
            self.initialized = True

        return success

    def update(self, image: np.ndarray) -> tuple[bool, Optional[List[float]]]:
        """Оновлення трекера"""
        if not self.initialized or self.cv_tracker is None:
            return False, None

        success, bbox_tuple = self.cv_tracker.update(image)

        if success:
            # Конвертація tuple -> list
            x, y, w, h = bbox_tuple
            bbox = [float(x), float(y), float(w), float(h)]
            self.current_bbox = bbox
            return True, bbox
        else:
            # Tracking failed - повертаємо останній відомий bbox
            return False, self.current_bbox

    def _create_tracker(self):
        """Створити OpenCV tracker (перевизначається у підкласах)"""
        raise NotImplementedError

    def reset(self):
        """Скидання стану"""
        super().reset()
        self.cv_tracker = None


@register_tracker
class KCFTracker(OpenCVTrackerBase):
    """
    KCF (Kernelized Correlation Filters) Tracker

    Переваги:
    - Швидкий
    - Добре працює з не-жорсткими деформаціями

    Недоліки:
    - Не дуже робастний до оклюзій
    - Не відновлюється після втрати
    """

    @classmethod
    def get_name(cls) -> str:
        return "KCF"

    def _create_tracker(self):
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            try:
                return cv2.TrackerKCF_create()
            except:
                raise RuntimeError("KCF tracker not available in this OpenCV version")


@register_tracker
class CSRTTracker(OpenCVTrackerBase):
    """
    CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)

    Переваги:
    - Дуже точний
    - Добре працює з оклюзіями
    - Адаптується до змін масштабу

    Недоліки:
    - Повільніший за KCF
    """

    @classmethod
    def get_name(cls) -> str:
        return "CSRT"

    def _create_tracker(self):
        try:
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            try:
                return cv2.TrackerCSRT_create()
            except:
                raise RuntimeError("CSRT tracker not available in this OpenCV version")


@register_tracker
class MedianFlowTracker(OpenCVTrackerBase):
    """
    Median Flow Tracker

    Переваги:
    - Добре виявляє помилки трекінгу
    - Робастний до шуму

    Недоліки:
    - Не працює з швидким рухом
    - Не відновлюється після втрати
    """

    @classmethod
    def get_name(cls) -> str:
        return "MedianFlow"

    def _create_tracker(self):
        # MedianFlow видалено з OpenCV 4.5.1+
        # Fallback на KCF
        try:
            return cv2.legacy.TrackerMedianFlow_create()
        except AttributeError:
            print("⚠️  MedianFlow не доступний, використовується KCF")
            return cv2.TrackerKCF_create()


@register_tracker
class MOSSETracker(OpenCVTrackerBase):
    """
    MOSSE (Minimum Output Sum of Squared Error) Tracker

    Переваги:
    - Найшвидший з усіх
    - Добре працює з змінами освітлення

    Недоліки:
    - Найменш точний
    - Не адаптується до змін масштабу
    """

    @classmethod
    def get_name(cls) -> str:
        return "MOSSE"

    def _create_tracker(self):
        # MOSSE також може бути у legacy
        try:
            return cv2.legacy.TrackerMOSSE_create()
        except AttributeError:
            try:
                return cv2.TrackerMOSSE_create()
            except:
                print("⚠️  MOSSE не доступний, використовується KCF")
                return cv2.TrackerKCF_create()


# Додаткові трекери (якщо доступні)

@register_tracker
class BoostingTracker(OpenCVTrackerBase):
    """Boosting Tracker (старий, повільний)"""

    @classmethod
    def get_name(cls) -> str:
        return "Boosting"

    def _create_tracker(self):
        try:
            return cv2.legacy.TrackerBoosting_create()
        except:
            return cv2.TrackerKCF_create()


@register_tracker
class MILTracker(OpenCVTrackerBase):
    """
    MIL (Multiple Instance Learning) Tracker

    Переваги:
    - Робастний до часткових оклюзій
    - Доступний у більшості версій OpenCV

    Недоліки:
    - Повільніший за KCF
    - Може дрейфувати при тривалому відстеженні
    """

    @classmethod
    def get_name(cls) -> str:
        return "MIL"

    def _create_tracker(self):
        try:
            return cv2.TrackerMIL_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerMIL_create()
            except:
                raise RuntimeError("MIL tracker not available in this OpenCV version")
