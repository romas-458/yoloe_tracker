"""
Trackers module - модульна система трекерів

Підтримує:
- FastSAM-based trackers
- OpenCV trackers (KCF, CSRT, etc.)
- Custom user trackers
"""

from .base_tracker import BaseTracker, TrackerRegistry
from .yoloe_vp_iou_tracker import YOLOeVPIoUTracker
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
    'KCFTracker',
    'CSRTTracker',
    'MedianFlowTracker',
    'MOSSETracker',
]
