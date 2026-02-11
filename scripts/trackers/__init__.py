"""
Trackers module - модульна система трекерів

Підтримує:
- FastSAM-based trackers
- OpenCV trackers (KCF, CSRT, etc.)
- Custom user trackers
"""

from .base_tracker import BaseTracker, TrackerRegistry
from .yoloe_vp_iou_tracker import YOLOeVPIoUTracker
from .yoloe_class_reinit_tracker import YOLOeClassReinitTracker
from .opencv_trackers import (
    KCFTracker,
    CSRTTracker,
    MedianFlowTracker,
    MOSSETracker
)

# Автоматична реєстрація трекерів
__all__ = [
    'BaseTracker',
    'TrackerRegistry',
    'YOLOeVPIoUTracker',
    'YOLOeClassReinitTracker',
    'KCFTracker',
    'CSRTTracker',
    'MedianFlowTracker',
    'MOSSETracker',
]
