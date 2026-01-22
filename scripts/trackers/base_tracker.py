"""
Base Tracker - абстрактний інтерфейс для всіх трекерів
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2


class BaseTracker(ABC):
    """
    Абстрактний базовий клас для трекерів

    Всі трекери мають реалізувати:
    - initialize() - ініціалізація з першим кадром
    - update() - оновлення на новому кадрі
    - get_name() - назва трекера
    """

    def __init__(self, **kwargs):
        """
        Ініціалізація трекера

        Args:
            **kwargs: Параметри специфічні для трекера
        """
        self.params = kwargs
        self.initialized = False
        self.current_bbox: Optional[List[float]] = None

    @abstractmethod
    def initialize(self, image: np.ndarray, bbox: List[float]) -> bool:
        """
        Ініціалізація трекера з першим кадром

        Args:
            image: Перше зображення (H, W, 3) BGR
            bbox: Початковий bbox [x, y, w, h]

        Returns:
            True якщо ініціалізація успішна
        """
        pass

    @abstractmethod
    def update(self, image: np.ndarray) -> tuple[bool, Optional[List[float]]]:
        """
        Оновлення трекера на новому кадрі

        Args:
            image: Нове зображення (H, W, 3) BGR

        Returns:
            (success, bbox) - чи успішно відстежено та bbox [x, y, w, h]
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Назва трекера

        Returns:
            Унікальна назва трекера (наприклад, "FastSAM-IoU", "KCF")
        """
        pass

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Параметри за замовчуванням для трекера

        Returns:
            Dict з параметрами
        """
        return {}

    def reset(self):
        """Скидання стану трекера"""
        self.initialized = False
        self.current_bbox = None


class TrackerRegistry:
    """
    Реєстр доступних трекерів

    Використовується для динамічного створення трекерів за назвою
    """

    _trackers: Dict[str, type] = {}

    @classmethod
    def register(cls, tracker_cls: type):
        """
        Реєстрація трекера

        Args:
            tracker_cls: Клас трекера (підклас BaseTracker)
        """
        if not issubclass(tracker_cls, BaseTracker):
            raise ValueError(f"{tracker_cls} must be subclass of BaseTracker")

        name = tracker_cls.get_name()
        cls._trackers[name] = tracker_cls

    @classmethod
    def get_tracker(cls, name: str, **kwargs) -> BaseTracker:
        """
        Створити інстанс трекера за назвою

        Args:
            name: Назва трекера
            **kwargs: Параметри для трекера

        Returns:
            Інстанс трекера

        Raises:
            ValueError: Якщо трекер не знайдено
        """
        if name not in cls._trackers:
            available = ", ".join(cls._trackers.keys())
            raise ValueError(f"Tracker '{name}' not found. Available: {available}")

        tracker_cls = cls._trackers[name]
        return tracker_cls(**kwargs)

    @classmethod
    def list_trackers(cls) -> List[str]:
        """Список доступних трекерів"""
        return list(cls._trackers.keys())

    @classmethod
    def get_tracker_info(cls, name: str) -> Dict[str, Any]:
        """Інформація про трекер"""
        if name not in cls._trackers:
            return {}

        tracker_cls = cls._trackers[name]
        return {
            'name': name,
            'class': tracker_cls.__name__,
            'default_params': tracker_cls.get_default_params(),
            'docstring': tracker_cls.__doc__
        }


def register_tracker(cls):
    """
    Декоратор для автоматичної реєстрації трекера

    Usage:
        @register_tracker
        class MyTracker(BaseTracker):
            ...
    """
    TrackerRegistry.register(cls)
    return cls
