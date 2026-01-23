"""
Modular Evaluation - універсальна оцінка для будь-яких трекерів

Підтримує:
- FastSAM-based trackers
- OpenCV trackers (KCF, CSRT, etc.)
- Custom user trackers

Використання:
    python modular_evaluation.py -d data/ -o results/ --tracker KCF
    python modular_evaluation.py -d data/ -o results/ --tracker FastSAM-IoU --num-frames 100
"""

import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import argparse
from tqdm import tqdm
import pandas as pd

# Імпорт трекерів
from trackers import TrackerRegistry, BaseTracker


@dataclass
class VideoResult:
    """Результат обробки одного відео"""
    tracker_name: str
    class_name: str
    video_name: str
    status: str  # 'success', 'failed', 'skipped'
    num_frames: int
    auc: float = 0.0
    precision_20: float = 0.0
    normalized_precision: float = 0.0  # Pnorm - нормалізована precision
    avg_iou: float = 0.0
    median_iou: float = 0.0
    success_0_5: float = 0.0
    tracking_rate: float = 0.0
    processing_time: float = 0.0
    fps: float = 0.0
    model_path: str = ""  # Шлях до моделі (якщо використовується)
    error_message: str = ""

    def to_dict(self):
        return asdict(self)


class ModularEvaluator:
    """Модульний evaluator для різних трекерів"""

    def __init__(self, tracker_name: str, tracker_params: Dict,
                 output_dir: Path, resume: bool = False):
        """
        Args:
            tracker_name: Назва трекера з registry
            tracker_params: Параметри для трекера
            output_dir: Папка для результатів
            resume: Продовжити з останньої точки
        """
        self.tracker_name = tracker_name
        self.tracker_params = tracker_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Перевірка доступності трекера
        available = TrackerRegistry.list_trackers()
        if tracker_name not in available:
            raise ValueError(f"Tracker '{tracker_name}' not available. "
                           f"Available: {', '.join(available)}")

        # Файли результатів
        self.results_file = self.output_dir / f"results_{tracker_name}.json"
        self.summary_file = self.output_dir / f"summary_{tracker_name}.json"

        # Результати
        self.results: List[VideoResult] = []
        self.processed_videos = set()

        if resume and self.results_file.exists():
            self._load_results()

    def _load_results(self):
        """Завантажити попередні результати"""
        with open(self.results_file, 'r') as f:
            data = json.load(f)
            for item in data:
                result = VideoResult(**item)
                self.results.append(result)
                if result.status == 'success':
                    video_id = f"{result.class_name}/{result.video_name}"
                    self.processed_videos.add(video_id)

        print(f"📂 Завантажено {len(self.results)} попередніх результатів")

    def _save_results(self):
        """Зберегти результати"""
        with open(self.results_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def load_test_list(self, test_list_file: Path) -> set:
        """
        Завантажити список відео для тестування з файлу

        Args:
            test_list_file: Шлях до файлу зі списком (формат: class-video)

        Returns:
            set - множина відео у форматі "class-video"
        """
        test_videos = set()
        if test_list_file.exists():
            with open(test_list_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        test_videos.add(line)
        return test_videos

    def find_sequences(self, data_dir: Path,
                      class_filter: Optional[str] = None,
                      video_filter: Optional[str] = None,
                      test_list: Optional[set] = None) -> List[tuple]:
        """Знайти всі послідовності"""
        sequences = []

        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_filter and class_name != class_filter:
                continue

            for video_dir in sorted(class_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                video_name = video_dir.name
                if video_filter and video_name != video_filter:
                    continue

                # Фільтр за test_list (якщо заданий)
                if test_list is not None:
                    # video_name вже містить клас (наприклад "airplane-1")
                    if video_name not in test_list:
                        continue

                img_dir = video_dir / "img"
                gt_file = video_dir / "groundtruth.txt"

                if img_dir.exists() and gt_file.exists():
                    sequences.append((class_name, video_name, video_dir))

        return sequences

    def load_groundtruth(self, gt_file: Path) -> List[Optional[List[float]]]:
        """Завантажити groundtruth"""
        bboxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or 'nan' in line.lower():
                    bboxes.append(None)
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        bbox = [float(p) for p in parts[:4]]
                        bboxes.append(bbox)
                    except:
                        bboxes.append(None)
                else:
                    bboxes.append(None)
        return bboxes

    def process_video(
        self,
        class_name: str,
        video_name: str,
        video_path: Path,
        num_frames: int = 0,
        visualize: bool = False,
        skip_if_exists: bool = True
    ) -> VideoResult:
        """
        Обробити одне відео

        Args:
            class_name: Назва класу
            video_name: Назва відео
            video_path: Шлях до відео
            num_frames: Кількість кадрів (0 = всі)
            visualize: Зберігати візуалізацію
            skip_if_exists: Пропустити якщо вже оброблено

        Returns:
            VideoResult
        """
        video_id = f"{class_name}/{video_name}"

        # Перевірка чи оброблено
        if skip_if_exists and video_id in self.processed_videos:
            for result in self.results:
                if result.class_name == class_name and result.video_name == video_name:
                    return result

            return VideoResult(
                tracker_name=self.tracker_name,
                class_name=class_name,
                video_name=video_name,
                status='skipped',
                num_frames=0
            )

        # Шляхи
        img_dir = video_path / "img"
        gt_file = video_path / "groundtruth.txt"

        # Завантаження даних
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(img_dir.glob(ext))
            image_files.extend(img_dir.glob(ext.upper()))
        image_files = sorted(image_files)

        if num_frames > 0:
            image_files = image_files[:num_frames]

        groundtruth = self.load_groundtruth(gt_file)[:len(image_files)]

        if not image_files or not groundtruth or groundtruth[0] is None:
            return VideoResult(
                tracker_name=self.tracker_name,
                class_name=class_name,
                video_name=video_name,
                status='failed',
                num_frames=0,
                error_message="No images or invalid groundtruth"
            )

        print(f"\n{'='*70}")
        model_info = f" | Модель: {Path(self.tracker_params.get('model_path', '')).name}" if self.tracker_params.get('model_path') else ""
        print(f"Трекер: {self.tracker_name}{model_info} | Відео: {class_name}/{video_name} ({len(image_files)} кадрів)")
        print(f"{'='*70}")

        start_time = time.time()

        try:
            # Створення трекера
            tracker = TrackerRegistry.get_tracker(self.tracker_name, **self.tracker_params)

            # Обробка
            results = self._run_tracking(
                tracker=tracker,
                image_files=image_files,
                groundtruth=groundtruth,
                video_path=video_path,
                visualize=visualize
            )

            elapsed = time.time() - start_time

            # Обчислення метрик
            metrics = self._compute_metrics(results, groundtruth)

            result = VideoResult(
                tracker_name=self.tracker_name,
                class_name=class_name,
                video_name=video_name,
                status='success',
                num_frames=len(image_files),
                auc=metrics['auc'],
                precision_20=metrics['precision_20'],
                normalized_precision=metrics['normalized_precision'],
                avg_iou=metrics['avg_iou'],
                median_iou=metrics['median_iou'],
                success_0_5=metrics['success_0.5'],
                tracking_rate=metrics['tracking_rate'],
                processing_time=elapsed,
                fps=len(image_files) / elapsed if elapsed > 0 else 0,
                model_path=self.tracker_params.get('model_path', '')
            )

            print(f"✅ AUC={result.auc:.3f} P@20={result.precision_20:.3f} "
                  f"Pnorm={result.normalized_precision:.3f} FPS={result.fps:.1f} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            result = VideoResult(
                tracker_name=self.tracker_name,
                class_name=class_name,
                video_name=video_name,
                status='failed',
                num_frames=0,
                processing_time=elapsed,
                model_path=self.tracker_params.get('model_path', ''),
                error_message=str(e)
            )
            print(f"❌ Помилка: {e}")

        return result

    def _run_tracking(
        self,
        tracker: BaseTracker,
        image_files: List[Path],
        groundtruth: List,
        video_path: Path,
        visualize: bool
    ) -> List[Optional[List[float]]]:
        """Запуск трекінгу"""
        results = []

        # Візуалізація
        vis_dir = None
        if visualize:
            vis_dir = self.output_dir / video_path.parent.name / video_path.name
            vis_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_path in enumerate(tqdm(image_files, desc="Tracking", leave=False)):
            image = cv2.imread(str(img_path))
            if image is None:
                results.append(None)
                continue

            if idx == 0:
                # Ініціалізація
                init_bbox = groundtruth[0]
                success = tracker.initialize(image, init_bbox)
                results.append(init_bbox if success else None)
            else:
                # Оновлення
                success, bbox = tracker.update(image)
                results.append(bbox if success else None)

            # Візуалізація
            if visualize and vis_dir:
                self._visualize_frame(image, results[-1], groundtruth[idx],
                                     vis_dir / f"{img_path.stem}.jpg", tracker)

        return results

    def _visualize_frame(self, image: np.ndarray,
                        pred_bbox: Optional[List[float]],
                        gt_bbox: Optional[List[float]],
                        output_path: Path,
                        tracker: Optional[BaseTracker] = None):
        """Візуалізація кадру"""
        # Отримати інформацію про трекінг (якщо доступно)
        tracking_info = None
        if tracker and hasattr(tracker, 'get_tracking_info'):
            tracking_info = tracker.get_tracking_info()

        # Додати інформацію про модель у верхній частині
        model_path = self.tracker_params.get('model_path', '')
        if model_path:
            # Отримати тільки назву файлу моделі
            model_name = Path(model_path).name
            text = f"Model: {model_name}"
            # Фон для тексту
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (5, 5), (15 + text_width, 10 + text_height), (0, 0, 0), -1)
            # Текст
            cv2.putText(image, text, (10, 5 + text_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Попередній bbox (сірий пунктир)
        if tracking_info and tracking_info.get('previous_bbox'):
            prev_bbox = tracking_info['previous_bbox']
            x, y, w, h = [int(v) for v in prev_bbox]
            # Пунктирний прямокутник
            self._draw_dashed_rectangle(image, (x, y), (x + w, y + h), (128, 128, 128), 1)
            cv2.putText(image, "Prev", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Топ-3 кандидати з IoU (різні кольори)
        if tracking_info and tracking_info.get('top_candidates'):
            colors = [
                (255, 200, 0),   # Блакитний - 1-й кандидат
                (255, 150, 0),   # Помаранчевий - 2-й кандидат
                (255, 100, 0)    # Темно-помаранчевий - 3-й кандидат
            ]
            for idx, candidate in enumerate(tracking_info['top_candidates']):
                bbox = candidate['bbox']
                iou = candidate.get('iou', 0.0)

                x, y, w, h = [int(v) for v in bbox]
                color = colors[idx] if idx < len(colors) else (200, 200, 200)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

                # Адаптивний текст в залежності від доступних даних
                text_parts = [f"C{idx+1}: IoU={iou:.2f}"]

                # Додати class_id якщо доступний (YOLOe-ClassReinit)
                if 'class_id' in candidate:
                    text_parts.append(f"cls={candidate['class_id']}")

                # Додати feature similarity якщо доступна (YOLOe-Feature)
                if 'feature_similarity' in candidate:
                    text_parts.append(f"feat={candidate['feature_similarity']:.2f}")

                # Додати score якщо доступний (YOLOe-Feature)
                if 'score' in candidate:
                    text_parts.append(f"sc={candidate['score']:.2f}")

                text = " ".join(text_parts)
                cv2.putText(image, text, (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Відкинуті кандидати з Фази 3 (червоний пунктир)
        if tracking_info and tracking_info.get('rejected_candidates'):
            rejected = tracking_info['rejected_candidates']
            for idx, candidate in enumerate(rejected):
                bbox = candidate['bbox']
                diou = candidate['diou']
                conf = candidate['conf']

                x, y, w, h = [int(v) for v in bbox]

                # Червоний пунктирний прямокутник
                self._draw_dashed_rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Текст з DIoU та conf
                text = f"REJECTED: DIoU={diou:.2f}, conf={conf:.2f}"
                cv2.putText(image, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Всі detections під час Phase 2/3 (search candidates) - жовтий/фіолетовий
        if tracking_info and tracking_info.get('search_candidates'):
            candidates = tracking_info['search_candidates']
            for idx, candidate in enumerate(candidates):
                bbox = candidate['bbox']
                conf = candidate['conf']
                diou = candidate['diou']
                phase = candidate.get('phase', 0)

                x, y, w, h = [int(v) for v in bbox]

                # Колір залежно від фази
                if phase == 2:
                    color = (0, 255, 255)  # Жовтий - Phase 2
                    phase_label = "P2"
                else:
                    color = (255, 0, 255)  # Фіолетовий - Phase 3
                    phase_label = "P3"

                # Тонка рамка
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

                # Текст з conf та DIoU
                text = f"{phase_label}: c={conf:.2f}"
                if diou > -float('inf'):
                    text += f" d={diou:.2f}"
                cv2.putText(image, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Last valid bbox (Фаза 2 - пошук/очікування) - помаранчевий
        if tracking_info and tracking_info.get('last_valid_bbox'):
            last_valid = tracking_info['last_valid_bbox']
            x, y, w, h = [int(v) for v in last_valid]
            lost_frames = tracking_info.get('lost_frames', 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Помаранчевий
            cv2.putText(image, f"Searching (lost={lost_frames})", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Predicted bbox (активний трекінг) - зелений
        if pred_bbox:
            x, y, w, h = [int(v) for v in pred_bbox]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Pred", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if gt_bbox:
            x, y, w, h = [int(v) for v in gt_bbox]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, "GT", (x, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Додаткова інформація про класи (якщо трекер підтримує)
        if tracking_info:
            y_offset = 30

            # Поточний клас
            if tracking_info.get('current_class') is not None:
                class_id = tracking_info['current_class']
                class_conf = tracking_info.get('current_class_conf', 0.0)
                text = f"Class: {class_id} ({class_conf:.2f})"
                cv2.putText(image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25

            # Набір класів об'єкта
            obj_classes = tracking_info.get('object_classes', {})
            if obj_classes:
                # Топ-3 класи
                top_classes = sorted(obj_classes.items(), key=lambda x: x[1], reverse=True)[:3]
                text = f"Classes: {', '.join([f'{cls}({cnt})' for cls, cnt in top_classes])}"
                cv2.putText(image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20

            # Режим фільтрування
            filter_mode = tracking_info.get('class_filter_mode', 'unknown')
            allowed = tracking_info.get('allowed_classes', [])
            if allowed:
                text = f"Filter: {filter_mode} -> {allowed}"
                cv2.putText(image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 20

            # Втрачені кадри
            lost = tracking_info.get('lost_frames', 0)
            if lost > 0:
                text = f"Lost: {lost} frames"
                cv2.putText(image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(str(output_path), image)

    @staticmethod
    def _draw_dashed_rectangle(image, pt1, pt2, color, thickness=1, dash_length=10):
        """Малювання пунктирного прямокутника"""
        x1, y1 = pt1
        x2, y2 = pt2

        # Верхня лінія
        for x in range(x1, x2, dash_length * 2):
            cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), color, thickness)

        # Нижня лінія
        for x in range(x1, x2, dash_length * 2):
            cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

        # Ліва лінія
        for y in range(y1, y2, dash_length * 2):
            cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)

        # Права лінія
        for y in range(y1, y2, dash_length * 2):
            cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

    def _compute_metrics(self, pred_bboxes: List, gt_bboxes: List) -> Dict:
        """Обчислити метрики"""
        ious = []
        distances = []
        norm_distances = []

        for pred, gt in zip(pred_bboxes, gt_bboxes):
            if pred and gt:
                iou = self._compute_iou(pred, gt)
                ious.append(iou)

                # Center distance
                pc = (pred[0] + pred[2]/2, pred[1] + pred[3]/2)
                gc = (gt[0] + gt[2]/2, gt[1] + gt[3]/2)
                dist = np.sqrt((pc[0]-gc[0])**2 + (pc[1]-gc[1])**2)
                distances.append(dist)

                # Normalized center distance (нормалізований розміром об'єкта)
                # Відповідає LaSOT normalized_center_error
                norm_dist = np.sqrt(
                    ((pc[0]-gc[0])/max(1.0, gt[2]))**2 +
                    ((pc[1]-gc[1])/max(1.0, gt[3]))**2
                )
                norm_distances.append(norm_dist)
            else:
                ious.append(0.0)
                distances.append(float('inf'))
                norm_distances.append(float('inf'))

        ious = np.array(ious)
        distances = np.array(distances)
        norm_distances = np.array(norm_distances)

        # Metrics
        metrics = {
            'auc': float(np.mean([np.mean(ious >= t) for t in np.arange(0, 1.01, 0.01)])),
            'precision_20': float(np.mean(distances <= 20.0)),
            'normalized_precision': float(np.mean([
                np.mean(norm_distances <= t) for t in np.linspace(0, 0.5, 51)
            ])),  # Pnorm - AUC normalized precision curve
            'avg_iou': float(np.mean(ious)),
            'median_iou': float(np.median(ious)),
            'success_0.5': float(np.mean(ious >= 0.5)),
            'tracking_rate': sum(1 for p in pred_bboxes if p) / len(pred_bboxes)
        }

        return metrics

    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """IoU"""
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

    def compute_aggregated_metrics(self) -> Dict:
        """Агрегація метрик"""
        successful = [r for r in self.results if r.status == 'success']

        if not successful:
            return {'per_video': [], 'per_class': {}, 'overall': {}}

        # Per-video
        per_video = [r.to_dict() for r in successful]

        # Per-class
        per_class = {}
        classes = set(r.class_name for r in successful)

        for class_name in classes:
            class_results = [r for r in successful if r.class_name == class_name]

            per_class[class_name] = {
                'num_videos': len(class_results),
                'avg_auc': np.mean([r.auc for r in class_results]),
                'avg_precision_20': np.mean([r.precision_20 for r in class_results]),
                'avg_normalized_precision': np.mean([r.normalized_precision for r in class_results]),
                'avg_iou': np.mean([r.avg_iou for r in class_results]),
                'avg_fps': np.mean([r.fps for r in class_results]),
            }

        # Overall
        overall = {
            'tracker': self.tracker_name,
            'model_path': self.tracker_params.get('model_path', ''),
            'num_videos': len(successful),
            'num_classes': len(classes),
            'avg_auc': np.mean([r.auc for r in successful]),
            'avg_precision_20': np.mean([r.precision_20 for r in successful]),
            'avg_normalized_precision': np.mean([r.normalized_precision for r in successful]),
            'avg_iou': np.mean([r.avg_iou for r in successful]),
            'avg_fps': np.mean([r.fps for r in successful]),
            'total_frames': sum(r.num_frames for r in successful),
            'total_time': sum(r.processing_time for r in successful),
            'failed': len([r for r in self.results if r.status == 'failed']),
        }

        return {
            'per_video': per_video,
            'per_class': per_class,
            'overall': overall
        }

    def generate_report(self):
        """Генерація звіту"""
        summary = self.compute_aggregated_metrics()

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        overall = summary['overall']
        print(f"\n{'='*70}")
        print(f"ЗВІТ: {self.tracker_name}")
        if self.tracker_params.get('model_path'):
            print(f"Модель: {Path(self.tracker_params.get('model_path')).name}")
        print(f"{'='*70}")

        # Перевірити чи є успішні результати
        if not overall:
            print("\n⚠️  Немає успішних результатів для звіту")
            print(f"   Failed: {len([r for r in self.results if r.status == 'failed'])}")
            print(f"\n📄 Результати: {self.results_file}")
            return

        print(f"\n📊 Overall ({overall['num_videos']} відео):")
        print(f"   AUC:           {overall['avg_auc']:.4f}")
        print(f"   Precision@20:  {overall['avg_precision_20']:.4f}")
        print(f"   Pnorm:         {overall['avg_normalized_precision']:.4f}")
        print(f"   Avg IoU:       {overall['avg_iou']:.4f}")
        print(f"   Avg FPS:       {overall['avg_fps']:.1f}")

        for class_name, metrics in sorted(summary['per_class'].items()):
            print(f"\n   {class_name}: AUC={metrics['avg_auc']:.3f} "
                  f"Pnorm={metrics['avg_normalized_precision']:.3f} FPS={metrics['avg_fps']:.1f}")

        print(f"\n📄 Результати: {self.results_file}")
        print(f"{'='*70}\n")

        # CSV
        if self.results:
            csv_file = self.output_dir / f"results_{self.tracker_name}.csv"
            df = pd.DataFrame([r.to_dict() for r in self.results if r.status == 'success'])
            df.to_csv(csv_file, index=False)

    def run_batch(
        self,
        data_dir: Path,
        num_frames: int = 0,
        class_filter: Optional[str] = None,
        video_filter: Optional[str] = None,
        test_list_file: Optional[Path] = None,
        max_videos: Optional[int] = None,
        visualize: bool = False
    ):
        """Запуск batch evaluation"""
        # Завантажити test list якщо заданий
        test_list = None
        if test_list_file:
            if not test_list_file.exists():
                print(f"❌ Test list файл не існує: {test_list_file}")
                return
            test_list = self.load_test_list(test_list_file)
            if test_list:
                print(f"📋 Завантажено test list: {len(test_list)} відео з {test_list_file.name}")
            else:
                print(f"⚠️  Test list файл порожній: {test_list_file}")
                return

        sequences = self.find_sequences(data_dir, class_filter, video_filter, test_list)

        if not sequences:
            print("❌ Послідовності не знайдено")
            return

        if max_videos:
            sequences = sequences[:max_videos]

        print(f"\n📂 Знайдено {len(sequences)} послідовностей")
        model_info = f" (Модель: {Path(self.tracker_params.get('model_path', '')).name})" if self.tracker_params.get('model_path') else ""
        print(f"🎯 Трекер: {self.tracker_name}{model_info}")

        for class_name, video_name, video_path in sequences:
            result = self.process_video(
                class_name=class_name,
                video_name=video_name,
                video_path=video_path,
                num_frames=num_frames,
                visualize=visualize,
                skip_if_exists=True
            )

            self.results.append(result)

            if result.status == 'success':
                self.processed_videos.add(f"{class_name}/{video_name}")

            self._save_results()

        self.generate_report()


def main():
    parser = argparse.ArgumentParser(
        description="Modular Evaluation - оцінка будь-яких трекерів",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:

  # Список доступних трекерів
  python modular_evaluation.py --list-trackers

  # KCF трекер
  python modular_evaluation.py -d data/ -o results/ --tracker KCF

  # FastSAM-IoU трекер, 100 кадрів
  python modular_evaluation.py -d data/ -o results/ --tracker FastSAM-IoU --num-frames 100

  # CSRT трекер, тільки airplane
  python modular_evaluation.py -d data/ -o results/ --tracker CSRT --class airplane

  # З візуалізацією
  python modular_evaluation.py -d data/ -o results/ --tracker KCF --visualize
        """
    )

    parser.add_argument('-d', '--data-dir', type=str,
                        help='Папка з даними')
    parser.add_argument('-o', '--output', type=str,
                        help='Папка для результатів')
    parser.add_argument('--tracker', type=str, default='KCF',
                        help='Назва трекера (KCF, CSRT, FastSAM-IoU, ...)')
    parser.add_argument('--list-trackers', action='store_true',
                        help='Показати список доступних трекерів')
    parser.add_argument('--num-frames', type=int, default=0,
                        help='Кількість кадрів (0 = всі)')
    parser.add_argument('--class', type=str, dest='class_filter',
                        help='Тільки вказаний клас')
    parser.add_argument('--video', type=str, dest='video_filter',
                        help='Тільки вказане відео (назва папки відео)')
    parser.add_argument('--test-list', type=str, dest='test_list_file',
                        help='Файл зі списком відео для тестування (формат: class-video, по рядку)')
    parser.add_argument('--max-videos', type=int,
                        help='Макс. відео')
    parser.add_argument('--visualize', action='store_true',
                        help='Зберігати візуалізацію')
    parser.add_argument('--resume', action='store_true',
                        help='Продовжити')

    # Параметри трекерів
    parser.add_argument('--tracker-params', type=str,
                        help='JSON з параметрами трекера, напр. \'{"process_noise": 2.0}\'')
    parser.add_argument('--model', type=str, default='yoloe-26s-seg-pf.pt',
                        help='Шлях до моделі (для YOLOe/FastSAM трекерів)')
    parser.add_argument('--imgsz', type=int, default=384,
                        help='[FastSAM/YOLOe] Розмір зображення')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='[FastSAM/YOLOe] Confidence')
    parser.add_argument('--iou-threshold', type=float, default=0.1,
                        help='[FastSAM/YOLOe] IoU threshold')

    args = parser.parse_args()

    # Список трекерів
    if args.list_trackers:
        print("\n📋 Доступні трекери:")
        for name in TrackerRegistry.list_trackers():
            info = TrackerRegistry.get_tracker_info(name)
            print(f"   - {name}")
            if info.get('default_params'):
                print(f"     Параметри: {info['default_params']}")
        return

    if not args.data_dir or not args.output:
        parser.print_help()
        return

    # Параметри трекера
    # Спочатку встановлюємо defaults з окремих параметрів командного рядка
    tracker_params = {}
    if args.tracker in ["FastSAM-IoU", "YOLOe-IoU", "YOLOe-Kalman", "YOLOe-ClassReinit", "YOLOe-Feature", "YOLOe-VP", "YOLOe-VP-Adaptive", "YOLOe-VP-IoU"]:
        tracker_params = {
            'model_path': args.model,
            'imgsz': args.imgsz,
            'conf': args.conf,
            'iou_threshold': args.iou_threshold,
        }

    # Потім перезаписуємо параметрами з --tracker-params (якщо є)
    # Це дає пріоритет --tracker-params над окремими аргументами
    if args.tracker_params:
        try:
            custom_params = json.loads(args.tracker_params)
            tracker_params.update(custom_params)  # Оновити існуючі параметри
        except json.JSONDecodeError as e:
            print(f"❌ Помилка парсингу --tracker-params: {e}")
            print(f"   Переконайтеся, що JSON правильний: {args.tracker_params}")
            return

    # Вивести фінальні параметри трекера
    if tracker_params:
        print(f"\n⚙️  Параметри трекера:")
        for key, value in tracker_params.items():
            print(f"   {key}: {value}")

    # Evaluator
    evaluator = ModularEvaluator(
        tracker_name=args.tracker,
        tracker_params=tracker_params,
        output_dir=Path(args.output),
        resume=args.resume
    )

    # Запуск
    # Обробити test_list_file якщо заданий
    test_list_file = None
    if args.test_list_file:
        test_list_file = Path(args.test_list_file)

    evaluator.run_batch(
        data_dir=Path(args.data_dir),
        num_frames=args.num_frames,
        class_filter=args.class_filter,
        video_filter=args.video_filter,
        test_list_file=test_list_file,
        max_videos=args.max_videos,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
