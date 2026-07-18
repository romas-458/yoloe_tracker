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

# Імпорт конфігу
from config_loader import ConfigLoader


@dataclass
class VideoResult:
    """Результат обробки одного відео"""
    tracker_name: str
    class_name: str
    video_name: str
    status: str  # 'success', 'failed', 'skipped'
    num_frames: int
    dataset: str = "lasot"  # 'lasot' or 'got10k'
    # LaSOT metrics
    auc: float = 0.0
    precision_20: float = 0.0
    normalized_precision: float = 0.0  # Pnorm - нормалізована precision
    avg_iou: float = 0.0
    median_iou: float = 0.0
    success_0_5: float = 0.0
    # GOT-10k metrics
    ao: float = 0.0  # Average Overlap
    sr_50: float = 0.0  # Success Rate @ 0.5
    sr_75: float = 0.0  # Success Rate @ 0.75
    # Common
    tracking_rate: float = 0.0
    processing_time: float = 0.0
    fps: float = 0.0
    repetitions: int = 1  # GOT-10k uses 3 repetitions
    model_path: str = ""  # Шлях до моделі (якщо використовується)
    error_message: str = ""

    def to_dict(self):
        return asdict(self)


class ModularEvaluator:
    """Модульний evaluator для різних трекерів"""

    def __init__(self, tracker_name: str, tracker_params: Dict,
                 output_dir: Path, dataset: str = 'lasot', resume: bool = False):
        """
        Args:
            tracker_name: Назва трекера з registry
            tracker_params: Параметри для трекера
            output_dir: Папка для результатів
            dataset: Датасет ('lasot' or 'got10k')
            resume: Продовжити з останньої точки
        """
        self.tracker_name = tracker_name
        self.tracker_params = tracker_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset.lower()

        if self.dataset not in ['lasot', 'got10k']:
            raise ValueError(f"Unknown dataset: {dataset}. Available: 'lasot', 'got10k'")

        # Перевірка доступності трекера
        available = TrackerRegistry.list_trackers()
        if tracker_name not in available:
            raise ValueError(f"Tracker '{tracker_name}' not available. "
                           f"Available: {', '.join(available)}")

        # Файли результатів
        self.results_file = self.output_dir / f"results_{tracker_name}.json"
        self.summary_file = self.output_dir / f"summary_{tracker_name}.json"

        # Буфер success/precision кривих (див. _stash_curves / _save_curves)
        self._curves: Dict[str, Dict] = {}
        self.dump_per_frame = False   # вмикається з CLI: --dump-per-frame

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
                        # Видалити inline коментарі (все після #)
                        video_name = line.split('#')[0].strip()
                        if video_name:
                            test_videos.add(video_name)
        return test_videos

    def find_sequences(self, data_dir: Path,
                      class_filter: Optional[str] = None,
                      video_filter: Optional[str] = None,
                      test_list: Optional[set] = None,
                      subset: str = 'val') -> List[tuple]:
        """Знайти всі послідовності"""
        if self.dataset == 'got10k':
            return self.find_got10k_sequences(data_dir, video_filter, test_list, subset)
        else:  # lasot
            return self.find_lasot_sequences(data_dir, class_filter, video_filter, test_list)

    def find_lasot_sequences(self, data_dir: Path,
                            class_filter: Optional[str] = None,
                            video_filter: Optional[str] = None,
                            test_list: Optional[set] = None) -> List[tuple]:
        """Знайти послідовності LaSOT"""
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

    def find_got10k_sequences(self, data_dir: Path,
                             seq_filter: Optional[str] = None,
                             test_list: Optional[set] = None,
                             subset: str = 'val') -> List[tuple]:
        """
        Знайти послідовності GOT-10k

        Структура:
        GOT-10k/
        ├── val/
        │   ├── GOT-10k_Val_000001/
        │   │   ├── 00000001.jpg
        │   │   └── groundtruth.txt
        │   └── GOT-10k_Val_000002/
        └── test/
        """
        sequences = []

        # Перевірити чи data_dir вже містить subset (val/test)
        subset_dir = data_dir / subset
        if not subset_dir.exists():
            # Можливо data_dir вже є subset папка
            if data_dir.name == subset or data_dir.name in ['val', 'test']:
                subset_dir = data_dir
            else:
                print(f"⚠️  Warning: subset '{subset}' not found in {data_dir}")
                return sequences

        for seq_dir in sorted(subset_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name

            # Фільтр за назвою послідовності
            if seq_filter and seq_name != seq_filter:
                continue

            # Фільтр за test_list
            if test_list is not None and seq_name not in test_list:
                continue

            gt_file = seq_dir / "groundtruth.txt"

            # Перевірити наявність кадрів (будь-який jpg файл)
            image_files = list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png"))

            if gt_file.exists() and len(image_files) > 0:
                # Для GOT-10k: class_name = subset, video_name = seq_name
                sequences.append((subset, seq_name, seq_dir))

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
        skip_if_exists: bool = True,
        repetitions: int = 1
    ) -> VideoResult:
        """
        Обробити одне відео

        Args:
            class_name: Назва класу (для LaSOT) або subset (для GOT-10k)
            video_name: Назва відео
            video_path: Шлях до відео
            num_frames: Кількість кадрів (0 = всі)
            visualize: Зберігати візуалізацію
            skip_if_exists: Пропустити якщо вже оброблено
            repetitions: Кількість повторів (GOT-10k використовує 3)

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
                num_frames=0,
                dataset=self.dataset
            )

        # Шляхи (різні для LaSOT та GOT-10k)
        if self.dataset == 'lasot':
            img_dir = video_path / "img"
        else:  # got10k
            img_dir = video_path  # Кадри безпосередньо в папці відео

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
                dataset=self.dataset,
                error_message="No images or invalid groundtruth"
            )

        print(f"\n{'='*70}")
        model_info = f" | Модель: {Path(self.tracker_params.get('model_path', '')).name}" if self.tracker_params.get('model_path') else ""
        dataset_info = f" | Dataset: {self.dataset.upper()}"
        print(f"Трекер: {self.tracker_name}{model_info}{dataset_info} | Відео: {class_name}/{video_name} ({len(image_files)} кадрів)")
        if repetitions > 1:
            print(f"Repetitions: {repetitions}")
        print(f"{'='*70}")

        start_time = time.time()

        try:
            # GOT-10k: виконати repetitions
            all_results = []
            for rep in range(repetitions):
                if repetitions > 1:
                    print(f"\n  Repetition {rep + 1}/{repetitions}")

                # Створення трекера
                tracker = TrackerRegistry.get_tracker(self.tracker_name, **self.tracker_params)

                # Обробка
                rep_results = self._run_tracking(
                    tracker=tracker,
                    image_files=image_files,
                    groundtruth=groundtruth,
                    video_path=video_path,
                    visualize=visualize and (rep == 0)  # Візуалізувати тільки перший раз
                )

                all_results.append(rep_results)

            elapsed = time.time() - start_time

            # Обчислення метрик
            if self.dataset == 'got10k':
                # GOT-10k: усереднити метрики по всіх repetitions
                all_metrics = [self._compute_got10k_metrics(results, groundtruth) for results in all_results]

                if self.dump_per_frame:
                    self._curves.setdefault(video_name, {})['ious'] = all_metrics[0]['_ious']

                metrics = {
                    'ao': np.mean([m['ao'] for m in all_metrics]),
                    'sr_50': np.mean([m['sr_50'] for m in all_metrics]),
                    'sr_75': np.mean([m['sr_75'] for m in all_metrics]),
                    'avg_iou': np.mean([m['avg_iou'] for m in all_metrics]),
                    'median_iou': np.median([m['median_iou'] for m in all_metrics]),
                    'tracking_rate': np.mean([m['tracking_rate'] for m in all_metrics]),
                }

                result = VideoResult(
                    tracker_name=self.tracker_name,
                    class_name=class_name,
                    video_name=video_name,
                    status='success',
                    num_frames=len(image_files),
                    dataset=self.dataset,
                    ao=metrics['ao'],
                    sr_50=metrics['sr_50'],
                    sr_75=metrics['sr_75'],
                    avg_iou=metrics['avg_iou'],
                    median_iou=metrics['median_iou'],
                    tracking_rate=metrics['tracking_rate'],
                    processing_time=elapsed,
                    fps=len(image_files) * repetitions / elapsed if elapsed > 0 else 0,
                    repetitions=repetitions,
                    model_path=self.tracker_params.get('model_path', '')
                )

                print(f"✅ AO={result.ao:.3f} SR0.5={result.sr_50:.3f} SR0.75={result.sr_75:.3f} "
                      f"FPS={result.fps:.1f} ({elapsed:.1f}s)")

            else:  # lasot
                # LaSOT: одна repetition
                results = all_results[0]
                metrics = self._compute_metrics(results, groundtruth)
                self._stash_curves(video_name, metrics)

                result = VideoResult(
                    tracker_name=self.tracker_name,
                    class_name=class_name,
                    video_name=video_name,
                    status='success',
                    num_frames=len(image_files),
                    dataset=self.dataset,
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
                dataset=self.dataset,
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
                # Текст-якір: якщо трекер підтримує, передати опис із nlp.txt
                if hasattr(tracker, "set_text_prompt"):
                    nlp_file = video_path / "nlp.txt"
                    if nlp_file.exists():
                        try:
                            tracker.set_text_prompt(nlp_file.read_text().strip())
                        except Exception:
                            pass
                # Ініціалізація
                init_bbox = groundtruth[0]
                success = tracker.initialize(image, init_bbox)
                results.append(init_bbox if success else None)
            else:
                # Оновлення
                success, bbox = tracker.update(image)
                results.append(bbox if success else None)

                # Debug інформація з proximity таблицею (для Phase 1)
                if hasattr(tracker, 'get_debug_info'):
                    debug_info = tracker.get_debug_info()
                    if debug_info.get('show_proximity_table') and debug_info.get('proximity_info'):
                        self._print_proximity_table(debug_info, idx)

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

        # Запам'ятати висоту першої таблиці для розміщення другої таблиці нижче
        first_table_height = 0

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
            cv2.putText(image, "Prevaaaaaa", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Кандидати з IoU (разні кольори для різних позицій)
        if tracking_info and tracking_info.get('top_candidates'):
            # Паліття кольорів для більш ніж 3 кандидатів
            colors = [
                (255, 200, 0),   # Блакитний - 1-й кандидат
                (255, 150, 0),   # Помаранчевий - 2-й кандидат
                (255, 100, 0),   # Темно-помаранчевий - 3-й кандидат
                (0, 165, 255),   # Помаранчевий (BGR) - 4-й
                (173, 255, 47),  # Зелено-жовтий - 5-й
                (240, 255, 240)  # Світло-блакитний - решта
            ]

            candidate_type = tracking_info['top_candidates'][0].get('type', 'standard') if tracking_info['top_candidates'] else 'standard'

            # Крок 1: Знайти best кандидата
            best_idx = None
            for idx, candidate in enumerate(tracking_info['top_candidates']):
                if candidate.get('is_best_match', False):
                    best_idx = idx
                    break

            # Крок 1a: Намалювати ббокси - спочатку non-best, потім best поверху
            for idx, candidate in enumerate(tracking_info['top_candidates']):
                if idx == best_idx:
                    continue  # Пропустити best - малюємо його потім

                bbox = candidate['bbox']
                x, y, w, h = [int(v) for v in bbox]
                color = colors[idx % len(colors)]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

            # Малюємо best ббокс поверху всіх (з товщиною 2)
            if best_idx is not None:
                candidate = tracking_info['top_candidates'][best_idx]
                bbox = candidate['bbox']
                x, y, w, h = [int(v) for v in bbox]
                color = colors[best_idx % len(colors)]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Крок 1b: Намалювати номери всередину ббоксу (для всіх у оригинальному порядку)
            for idx, candidate in enumerate(tracking_info['top_candidates']):
                bbox = candidate['bbox']
                x, y, w, h = [int(v) for v in bbox]
                color = colors[idx % len(colors)]  # Визначити колір для цього кандидата

                # Намалювати номер всередину ббоксу в лівому нижньому куті
                text_number = f"{idx+1}"
                font_scale = 0.4
                font_thickness = 2
                text_size = cv2.getTextSize(text_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

                # Позиція: лівий нижній кут ббоксу з невеликим зміщенням
                text_x = x + 5
                text_y = y + h - 5

                # Малий фон для номера
                cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 2),
                             (text_x + text_size[0] + 2, text_y + 2), color, -1)
                cv2.putText(image, text_number, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            # Крок 2: Створити таблицю з інформацією у верхньому лівому куті
            candidates = tracking_info['top_candidates']

            if candidate_type == 'samurai':
                # Таблиця для SAMURAI: # | IoU | aff | hyb | conf | status
                table_header = "# | IoU    | aff    | hyb    | conf   |"
                table_lines = [table_header, "-" * len(table_header)]

                for idx, candidate in enumerate(candidates):
                    iou = candidate.get('iou', 0.0)
                    affinity = candidate.get('affinity', 0.0)
                    hybrid = candidate.get('hybrid', 0.0)
                    # Ключ може бути 'conf' або 'confidence' залежно від трекера
                    conf = candidate.get('conf', candidate.get('confidence', 0.0))
                    is_best = candidate.get('is_best_match', False)

                    status = "BEST" if is_best else ""
                    line = f"{idx+1} | {iou:.3f} | {affinity:.3f} | {hybrid:.3f} | {conf:.3f} | {status}"
                    table_lines.append(line)
            elif candidate_type in ['phase2', 'phase3']:
                # Таблиця для Phase 2/3: # | Phase | DIoU | conf | status
                table_header = "# | Phase | DIoU   | Conf   | Status"
                table_lines = [table_header, "-" * len(table_header)]

                for idx, candidate in enumerate(candidates):
                    phase_label = "P2" if candidate.get('phase', 0) == 2 else "P3"
                    diou = candidate.get('diou', 0.0)
                    # Ключ може бути 'conf' або 'confidence' залежно від трекера
                    conf = candidate.get('conf', candidate.get('confidence', 0.0))
                    status = candidate.get('status', 'unknown')
                    is_best = candidate.get('is_best_match', False)
                    status_text = "ACCEPT" if status == 'accepted' else "REJECT"
                    best_marker = " BEST" if is_best else ""

                    line = f"{idx+1} | {phase_label}  | {diou:.3f} | {conf:.3f} | {status_text}{best_marker}"
                    table_lines.append(line)
            else:
                # Таблиця для standard: # | IoU | conf | size | status
                table_header = "# | IoU    | conf   | size | status"
                table_lines = [table_header, "-" * len(table_header)]

                for idx, candidate in enumerate(candidates):
                    iou = candidate.get('iou', 0.0)
                    # Ключ може бути 'conf' або 'confidence' залежно від трекера
                    conf = candidate.get('conf', candidate.get('confidence', 0.0))
                    size_valid = candidate.get('size_valid', True)
                    is_best = candidate.get('is_best_match', False)

                    size_status = "OK" if size_valid else "SMALL"
                    status = "BEST" if is_best else ""
                    line = f"{idx+1} | {iou:.3f} | {conf:.3f} | {size_status} | {status}"
                    table_lines.append(line)

            # Намалювати таблицю в лівому верхньому куті
            font_size = 0.35
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_height = 18

            # Фон для таблиці
            table_height = len(table_lines) * line_height + 10
            first_table_height = table_height  # Запам'ятати висоту для другої таблиці
            # Ширина за найширшим рядком (текст малюється від x=10)
            table_text_w = max(cv2.getTextSize(l, font, font_size, font_thickness)[0][0]
                               for l in table_lines)
            table_x2 = 10 + table_text_w + 8
            cv2.rectangle(image, (5, 5), (table_x2, 5 + table_height), (30, 30, 30), -1)
            cv2.rectangle(image, (5, 5), (table_x2, 5 + table_height), (200, 200, 200), 1)

            # Малювати рядки таблиці
            for line_idx, line in enumerate(table_lines):
                y_pos = 20 + line_idx * line_height

                # Заголовок та розділювач - білий, інші рядки - сірий
                if line_idx < 2:
                    text_color = (255, 255, 255)
                else:
                    text_color = (200, 200, 200)

                cv2.putText(image, line, (10, y_pos),
                           font, font_size, text_color, font_thickness)

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

            # Таблиця для Phase 2/3 search candidates з статусом прийняття/відкидання
            if len(candidates) > 0:
                phase = candidates[0].get('phase', 0)

                # Формування таблиці
                table_header = "# | Phase | DIoU   | Conf   | Status"
                table_lines = [table_header, "-" * len(table_header)]

                for idx, candidate in enumerate(candidates):
                    phase_label = "P2" if candidate.get('phase', 0) == 2 else "P3"
                    diou_val = candidate.get('diou', -float('inf'))
                    # Ключ може бути 'conf' або 'confidence' залежно від трекера
                    conf_val = candidate.get('conf', candidate.get('confidence', 0.0))
                    status = candidate.get('status', 'unknown')
                    status_text = "ACCEPT" if status == 'accepted' else "REJECT"

                    line = f"{idx+1} | {phase_label}  | {diou_val:.3f} | {conf_val:.3f} | {status_text}"
                    table_lines.append(line)

                # Малювати таблицю нижче першої таблиці (top_candidates)
                font_size = 0.35
                font_thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                line_height = 18

                # Фон для таблиці
                table_height = len(table_lines) * line_height + 10

                # Обчислити Y позицію нижче першої таблиці
                # first_table_height запам'ятана при малюванні першої таблиці
                zazor = 10  # Невеликий зазор між таблицями
                table_y1 = 5 + first_table_height + zazor
                table_y2 = table_y1 + table_height

                # Ширина за найширшим рядком (текст малюється від x=10)
                table_text_w = max(cv2.getTextSize(l, font, font_size, font_thickness)[0][0]
                                   for l in table_lines)
                table_x2 = 10 + table_text_w + 8
                cv2.rectangle(image, (5, table_y1), (table_x2, table_y2), (30, 30, 30), -1)
                cv2.rectangle(image, (5, table_y1), (table_x2, table_y2), (200, 200, 200), 1)

                # Малювати рядки таблиці
                for line_idx, line in enumerate(table_lines):
                    y_pos = table_y1 + 15 + line_idx * line_height

                    # Заголовок та розділювач - білий, інші рядки - сірий
                    if line_idx < 2:
                        text_color = (255, 255, 255)
                    else:
                        text_color = (200, 200, 200)

                    cv2.putText(image, line, (10, y_pos),
                               font, font_size, text_color, font_thickness)

        # Last valid bbox (Phase 2 і Phase 3 - об'єкт втрачено) - помаранчевий
        if tracking_info and tracking_info.get('last_valid_bbox'):
            last_valid = tracking_info['last_valid_bbox']
            x, y, w, h = [int(v) for v in last_valid]
            lost_frames = tracking_info.get('lost_frames', 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Помаранчевий
            cv2.putText(image, f"Last Valid (lost={lost_frames})", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Phase 3 reference bbox (bbox з яким порівнюються кандидати) - жовтий пунктир
        if tracking_info and tracking_info.get('phase3_reference_bbox'):
            ref = tracking_info['phase3_reference_bbox']
            ref_mode = tracking_info.get('phase3_ref_mode', '')
            x, y, w, h = [int(v) for v in ref]
            # Малювати тільки якщо відрізняється від last_valid (інакше зайве накладання)
            last_valid = tracking_info.get('last_valid_bbox')
            is_same_as_last_valid = (last_valid is not None and
                                     abs(last_valid[0] - ref[0]) < 2 and
                                     abs(last_valid[1] - ref[1]) < 2)
            if not is_same_as_last_valid:
                self._draw_dashed_rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Жовтий
                cv2.putText(image, f"Ref ({ref_mode})", (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Predicted bbox (активний трекінг)
        if pred_bbox:
            x, y, w, h = [int(v) for v in pred_bbox]

            # Перевірити чи це Kalman-only bbox (не валідовано детекціями)
            is_kalman_only = tracking_info.get('last_bbox_is_kalman_only', False) if tracking_info else False

            if is_kalman_only:
                # Фіолетовий пунктирний для Kalman-only
                color = (255, 0, 255)  # Фіолетовий
                thickness = 2
                self._draw_dashed_rectangle(image, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(image, "Kalman Only", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Зелений для валідованих детекцій
                color = (0, 255, 0)  # Зелений
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, "Pred", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if gt_bbox:
            x, y, w, h = [int(v) for v in gt_bbox]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, "GT", (x, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Додаткова інформація про класи (якщо трекер підтримує)
        if tracking_info:
            # Розміщувати цю інформацію нижче таблиць, щоб уникнути накладання
            # Обчислити Y початку на основі висоти таблиць
            zazor_after_tables = 20

            if first_table_height > 0:
                # Якщо є таблиці, розміщувати нижче них
                y_offset = 5 + first_table_height + zazor_after_tables

                # Якщо є друга таблиця (search_candidates), розміщувати ще нижче
                if tracking_info.get('search_candidates'):
                    # Друга таблиця також має висоту, приблизно 100-150px
                    # Додаємо додатковий зазор
                    y_offset += 150
            else:
                # Якщо немає таблиць, розміщувати в верхньому куті
                y_offset = 30

            # Поточний клас - використовувати confidence з best candidate у таблиці
            if tracking_info.get('current_class') is not None:
                class_id = tracking_info['current_class']
                class_conf = tracking_info.get('current_class_conf', 0.0)

                # Якщо є top_candidates, використовувати confidence з BEST candidate
                if tracking_info.get('top_candidates'):
                    for candidate in tracking_info['top_candidates']:
                        if candidate.get('is_best_match', False):
                            class_conf = candidate.get('confidence', class_conf)
                            break

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

        # Кандидати multi-VPE: рамка + мітка виду пам'яті (anchor/LT/ST), що її зловив.
        # Колір за видом (BGR): anchor синій, LT зелений, ST помаранчевий.
        if tracking_info and tracking_info.get('multi_vpe_candidates'):
            view_colors = {'anchor': (208, 111, 42), 'LT': (104, 144, 15), 'ST': (52, 106, 235)}
            H = image.shape[0]
            for cand in tracking_info['multi_vpe_candidates']:
                x, y, w, h = [int(v) for v in cand['bbox']]
                view = cand['view']
                color = view_colors.get(view, (200, 200, 200))
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                label = f"{view} {cand['conf']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                ly = y + h + th + 4 if y - 6 < th else y - 6
                cv2.rectangle(image, (x, ly - th - 3), (x + tw + 4, ly + 2), color, -1)
                cv2.putText(image, label, (x + 2, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            # легенда видів у нижньому лівому куті
            lx, ly = 10, H - 10
            cv2.putText(image, "multi-VPE:", (lx, ly - 3 * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for i, (v, c) in enumerate(view_colors.items()):
                yy = ly - (2 - i) * 18
                cv2.rectangle(image, (lx, yy - 10), (lx + 14, yy), c, -1)
                cv2.putText(image, v, (lx + 20, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # ⭐ Пул tracklet-ів виключення за співіснуванням (прапорець coexist_draw_ids):
        # блакитний = живий кандидат, червоний = проштампований «не ціль» (вето на
        # re-detection). Число в дужках — лічильник роз'єднаного доказу до порога 30.
        if (tracking_info and tracking_info.get('coexist_tracklets')
                and self.tracker_params.get('coexist_draw_ids', False)):
            for t in tracking_info['coexist_tracklets']:
                x, y, w, h = [int(v) for v in t['bbox']]
                stamped = t['stamped']
                color = (60, 60, 230) if stamped else (230, 180, 40)  # BGR: черв / блакит
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2 if stamped else 1)
                tag = f"#{t['id']}"
                if stamped:
                    tag += " NIET"
                elif t['disjoint'] > 0:
                    tag += f" ({t['disjoint']}/30)"
                cv2.putText(image, tag, (x, y + h + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            lx, ly = 10, image.shape[0] - 10
            cv2.putText(image, "coexist:", (lx, ly - 2 * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for i, (lbl, c) in enumerate([("live", (230, 180, 40)),
                                          ("NIET (vetoed)", (60, 60, 230))]):
                yy = ly - (1 - i) * 18
                cv2.rectangle(image, (lx, yy - 10), (lx + 14, yy), c, -1)
                cv2.putText(image, lbl, (lx + 20, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

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

    def _stash_curves(self, video_name: str, metrics: Dict):
        """
        Забрати success/precision криві (і, за прапорцем, per-frame IoU) з metrics
        у буфер, а самі приватні ключі видалити — щоб не потрапили у VideoResult.

        Криві дешеві: 101+51+51 float на відео. Per-frame IoU важчий, тому лише
        за self.dump_per_frame. Без цих даних неможливі success plot, precision
        plot і часовий профіль IoU — раніше вони рахувались і викидались.
        """
        buf = self._curves.setdefault(video_name, {})
        buf['success'] = metrics.pop('_success_curve')
        buf['precision'] = metrics.pop('_precision_curve')
        buf['norm_precision'] = metrics.pop('_norm_prec_curve')
        ious = metrics.pop('_ious')
        if self.dump_per_frame:
            buf['ious'] = ious

    def _save_curves(self):
        """Записати буфер кривих у <output_dir>/curves_<tracker>.npz"""
        if not self._curves:
            return
        flat = {}
        for video, d in self._curves.items():
            for key, arr in d.items():
                flat[f'{video}|{key}'] = arr
        path = self.output_dir / f'curves_{self.tracker_name}.npz'
        np.savez_compressed(path, **flat)
        print(f"📈 Криві збережено: {path} ({len(self._curves)} відео)")

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

        # Пороги success/precision plot-ів (конвенція OTB/LaSOT)
        succ_thr = np.arange(0, 1.01, 0.01)          # 101 точка
        prec_thr = np.arange(0, 51, 1.0)             # 51 точка, пікселі
        nprec_thr = np.linspace(0, 0.5, 51)          # 51 точка, нормалізовані

        # Криві. AUC та Pnorm — це просто середні по цих кривих, тож раніше вони
        # рахувались і викидались. Зберігаємо, бо без них немає success plot.
        success_curve = np.array([np.mean(ious >= t) for t in succ_thr])
        precision_curve = np.array([np.mean(distances <= t) for t in prec_thr])
        norm_prec_curve = np.array([np.mean(norm_distances <= t) for t in nprec_thr])

        # Metrics
        metrics = {
            'auc': float(success_curve.mean()),
            'precision_20': float(np.mean(distances <= 20.0)),
            'normalized_precision': float(norm_prec_curve.mean()),  # Pnorm
            'avg_iou': float(np.mean(ious)),
            'median_iou': float(np.median(ious)),
            'success_0.5': float(np.mean(ious >= 0.5)),
            'tracking_rate': sum(1 for p in pred_bboxes if p) / len(pred_bboxes),
            # приватні поля: не потрапляють у VideoResult, забираються викликачем
            '_success_curve': success_curve,
            '_precision_curve': precision_curve,
            '_norm_prec_curve': norm_prec_curve,
            '_ious': ious.astype(np.float32),
        }

        return metrics

    def _print_proximity_table(self, debug_info: Dict, frame_idx: int):
        """
        Вивести таблицю proximity до anchor та memory для Phase 1

        Args:
            debug_info: Dict з результатом get_debug_info()
            frame_idx: Індекс кадру
        """
        try:
            proximity = debug_info.get('proximity_info', {})
            metadata = proximity.get('metadata', {})

            print(f"\n{'='*100}")
            print(f"🔍 PHASE 1 PROXIMITY DEBUG TABLE [Frame {frame_idx}]")
            print(f"{'='*100}")

            # Anchor VPE інформація
            anchor_info = metadata.get('anchor', {})
            print(f"\n📌 ANCHOR VPE:")
            print(f"   Available: {anchor_info.get('available')}")
            if anchor_info.get('available'):
                conf_val = anchor_info.get('conf')
                print(f"   Conf: {conf_val:.3f}" if conf_val is not None else f"   Conf: N/A")
                print(f"   Frame: {anchor_info.get('frame')}")

            anchor_prox = proximity.get('anchor_proximity')
            prox_str = f"{anchor_prox:.3f}" if anchor_prox is not None else 'N/A'
            print(f"   Proximity: {prox_str}")

            # Long-term memory інформація
            lt_info = metadata.get('long_term', {})
            print(f"\n📌 LONG-TERM MEMORY:")
            print(f"   Count: {lt_info.get('count')}/{lt_info.get('capacity')}")
            if lt_info.get('confs'):
                print(f"   Confs: {[f'{c:.3f}' for c in lt_info.get('confs', [])]}")
                print(f"   Frames: {lt_info.get('frames', [])}")

            lt_prox = proximity.get('lt_proximity')
            prox_str = f"{lt_prox:.3f}" if lt_prox is not None else 'N/A'
            print(f"   Proximity: {prox_str}")

            # Short-term memory інформація
            st_info = metadata.get('short_term', {})
            print(f"\n📌 SHORT-TERM MEMORY:")
            print(f"   Count: {st_info.get('count')}/{st_info.get('capacity')}")
            if st_info.get('confs'):
                print(f"   Confs: {[f'{c:.3f}' for c in st_info.get('confs', [])]}")
                print(f"   Ages: {st_info.get('ages', [])}")
                print(f"   Frames: {st_info.get('frames', [])}")

            st_prox = proximity.get('st_proximity')
            prox_str = f"{st_prox:.3f}" if st_prox is not None else 'N/A'
            print(f"   Proximity: {prox_str}")

            # Додати таблицю для всіх детекцій на кадрі
            detections = proximity.get('detections', [])
            selected_idx = proximity.get('selected_detection_idx')
            if detections:
                print(f"\n📊 ALL DETECTIONS PROXIMITY (Top-7 + Tracked):")
                print(f"   {'Idx':<4} {'Conf':<7} {'IoU':<7} {'Anchor':<10} {'LT':<10} {'ST':<10} {'Status':<10}")
                print(f"   {'-'*75}")
                for det in detections:
                    idx = det.get('idx', '?')
                    conf = det.get('conf', 0)
                    iou = det.get('iou', 0)
                    anchor_p = det.get('anchor_proximity')
                    lt_p = det.get('lt_proximity')
                    st_p = det.get('st_proximity')

                    anchor_str = f"{anchor_p:.3f}" if anchor_p is not None else 'N/A'
                    lt_str = f"{lt_p:.3f}" if lt_p is not None else 'N/A'
                    st_str = f"{st_p:.3f}" if st_p is not None else 'N/A'

                    # Позначити вибрану детекцію
                    status = "✅ TRACKED" if idx == selected_idx else ""

                    print(f"   {idx:<4} {conf:<7.3f} {iou:<7.3f} {anchor_str:<10} {lt_str:<10} {st_str:<10} {status:<10}")

            print(f"\n{'='*100}\n")

        except Exception as e:
            print(f"⚠️  Error printing proximity table: {e}")

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

    def _compute_got10k_metrics(self, pred_bboxes: List, gt_bboxes: List) -> Dict:
        """
        Обчислити GOT-10k метрики: AO, SR0.5, SR0.75

        GOT-10k метрики:
        - AO (Average Overlap): середній IoU
        - SR0.5: Success Rate @ IoU >= 0.5
        - SR0.75: Success Rate @ IoU >= 0.75
        """
        ious = []

        for pred, gt in zip(pred_bboxes, gt_bboxes):
            if pred and gt:
                iou = self._compute_iou(pred, gt)
                ious.append(iou)
            else:
                ious.append(0.0)

        ious = np.array(ious)

        # GOT-10k metrics
        metrics = {
            'ao': float(np.mean(ious)),  # Average Overlap
            'sr_50': float(np.mean(ious >= 0.5)),  # Success Rate @ 0.5
            'sr_75': float(np.mean(ious >= 0.75)),  # Success Rate @ 0.75
            'avg_iou': float(np.mean(ious)),  # Alias for AO
            'median_iou': float(np.median(ious)),
            'tracking_rate': sum(1 for p in pred_bboxes if p) / len(pred_bboxes),
            # Success curve for GOT-10k (101 thresholds from 0 to 1)
            'succ_curve': [float(np.mean(ious >= t)) for t in np.linspace(0, 1, 101)],
            '_ious': ious.astype(np.float32),   # приватне: для --dump-per-frame
        }

        return metrics

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
                'avg_iou': np.mean([r.avg_iou for r in class_results]),
                'avg_fps': np.mean([r.fps for r in class_results]),
            }

            # Dataset-specific metrics
            if self.dataset == 'got10k':
                per_class[class_name]['avg_ao'] = np.mean([r.ao for r in class_results])
                per_class[class_name]['avg_sr_50'] = np.mean([r.sr_50 for r in class_results])
                per_class[class_name]['avg_sr_75'] = np.mean([r.sr_75 for r in class_results])
            else:  # lasot
                per_class[class_name]['avg_auc'] = np.mean([r.auc for r in class_results])
                per_class[class_name]['avg_precision_20'] = np.mean([r.precision_20 for r in class_results])
                per_class[class_name]['avg_normalized_precision'] = np.mean([r.normalized_precision for r in class_results])

        # Overall
        overall = {
            'tracker': self.tracker_name,
            'dataset': self.dataset,
            'model_path': self.tracker_params.get('model_path', ''),
            'num_videos': len(successful),
            'num_classes': len(classes),
            'avg_iou': np.mean([r.avg_iou for r in successful]),
            'avg_fps': np.mean([r.fps for r in successful]),
            'total_frames': sum(r.num_frames for r in successful),
            'total_time': sum(r.processing_time for r in successful),
            'failed': len([r for r in self.results if r.status == 'failed']),
        }

        # Dataset-specific metrics
        if self.dataset == 'got10k':
            overall['avg_ao'] = np.mean([r.ao for r in successful])
            overall['avg_sr_50'] = np.mean([r.sr_50 for r in successful])
            overall['avg_sr_75'] = np.mean([r.sr_75 for r in successful])
        else:  # lasot
            overall['avg_auc'] = np.mean([r.auc for r in successful])
            overall['avg_precision_20'] = np.mean([r.precision_20 for r in successful])
            overall['avg_normalized_precision'] = np.mean([r.normalized_precision for r in successful])

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

        self._save_curves()

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

        print(f"\n📊 Overall ({overall['num_videos']} відео | Dataset: {self.dataset.upper()}):")
        if self.dataset == 'got10k':
            print(f"   AO (Avg Overlap):  {overall['avg_ao']:.4f}")
            print(f"   SR@0.5:            {overall['avg_sr_50']:.4f}")
            print(f"   SR@0.75:           {overall['avg_sr_75']:.4f}")
        else:  # lasot
            print(f"   AUC:               {overall['avg_auc']:.4f}")
            print(f"   Precision@20:      {overall['avg_precision_20']:.4f}")
            print(f"   Pnorm:             {overall['avg_normalized_precision']:.4f}")
        print(f"   Avg IoU:           {overall['avg_iou']:.4f}")
        print(f"   Avg FPS:           {overall['avg_fps']:.1f}")

        for class_name, metrics in sorted(summary['per_class'].items()):
            if self.dataset == 'got10k':
                print(f"\n   {class_name}: AO={metrics['avg_ao']:.3f} "
                      f"SR0.5={metrics['avg_sr_50']:.3f} SR0.75={metrics['avg_sr_75']:.3f} FPS={metrics['avg_fps']:.1f}")
            else:  # lasot
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
        visualize: bool = False,
        subset: str = 'val',
        repetitions: int = None
    ):
        """Запуск batch evaluation"""
        # Auto-detect repetitions for GOT-10k
        if repetitions is None:
            repetitions = 3 if self.dataset == 'got10k' else 1

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

        sequences = self.find_sequences(data_dir, class_filter, video_filter, test_list, subset)

        if not sequences:
            print("❌ Послідовності не знайдено")
            return

        if max_videos:
            sequences = sequences[:max_videos]

        print(f"\n📂 Знайдено {len(sequences)} послідовностей (Dataset: {self.dataset.upper()})")
        if self.dataset == 'got10k':
            print(f"   Subset: {subset}, Repetitions: {repetitions}")
        model_info = f" (Модель: {Path(self.tracker_params.get('model_path', '')).name})" if self.tracker_params.get('model_path') else ""
        print(f"🎯 Трекер: {self.tracker_name}{model_info}")

        for class_name, video_name, video_path in sequences:
            result = self.process_video(
                class_name=class_name,
                video_name=video_name,
                video_path=video_path,
                num_frames=num_frames,
                visualize=visualize,
                skip_if_exists=True,
                repetitions=repetitions
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

  # LaSOT - KCF трекер
  python modular_evaluation.py -d data/LaSOT/ -o results/ --tracker KCF

  # LaSOT - FastSAM-IoU трекер, 100 кадрів
  python modular_evaluation.py -d data/LaSOT/ -o results/ --tracker FastSAM-IoU --num-frames 100

  # LaSOT - CSRT трекер, тільки airplane
  python modular_evaluation.py -d data/LaSOT/ -o results/ --tracker CSRT --class airplane

  # GOT-10k - KCF на одному відео
  python modular_evaluation.py -d data/GOT-10k/ -o results/ --tracker KCF --dataset got10k --video GOT-10k_Val_000001

  # GOT-10k - YOLOe на перших 10 відео val
  python modular_evaluation.py -d data/GOT-10k/ -o results/ --tracker YOLOe-VP-IoU --dataset got10k --first 10

  # GOT-10k - CSRT з 1 repetition (для швидкого тесту)
  python modular_evaluation.py -d data/GOT-10k/ -o results/ --tracker CSRT --dataset got10k --first 3 --repetitions 1
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

    # Dataset options
    parser.add_argument('--dataset', type=str, default='lasot', choices=['lasot', 'got10k'],
                        help='Dataset (lasot or got10k, default: lasot)')
    parser.add_argument('--subset', type=str, default='val', choices=['val', 'test'],
                        help='GOT-10k subset (val or test, default: val)')
    parser.add_argument('--repetitions', type=int,
                        help='Number of repetitions (GOT-10k default: 3, LaSOT default: 1)')
    parser.add_argument('--dump-per-frame', action='store_true',
                        help='Зберегти ще й per-frame IoU у curves_*.npz (для часових профілів). '
                             'Success/precision криві зберігаються завжди.')
    parser.add_argument('--first', type=int,
                        help='Evaluate first N sequences (for quick testing)')

    # Параметри трекерів
    parser.add_argument('--tracker-params', type=str,
                        help='JSON з параметрами трекера, напр. \'{"process_noise": 2.0}\'')
    parser.add_argument('--tracker-config', type=str,
                        help='Шлях до файлу з параметрами трекера (YAML/JSON, напр. configs/yoloe-vp-iou.yaml)')
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

    # Завантажити параметри з файлу конфігу (--tracker-config)
    if args.tracker_config:
        try:
            config_loader = ConfigLoader()
            config_params = config_loader.load(args.tracker_config)
            tracker_params.update(config_params)

            # Завантажити метадані для інформативного виводу
            try:
                metadata = config_loader.load_metadata(args.tracker_config)
                if metadata.name:
                    print(f"📄 Конфіг: {metadata.name}")
                if metadata.description:
                    print(f"   {metadata.description}")
            except:
                pass

        except Exception as e:
            print(f"❌ Помилка завантаження конфігу: {e}")
            return

    # Потім перезаписуємо параметрами з --tracker-params (якщо є)
    # Це дає пріоритет --tracker-params над конфігом та окремими аргументами
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

    # Handle --first parameter
    max_videos = args.max_videos or args.first

    # Визначити ім'я трекера: з конфігу (пріоритет) або з аргументу --tracker
    tracker_name = tracker_params.get('tracker', args.tracker) if tracker_params else args.tracker

    # Evaluator
    evaluator = ModularEvaluator(
        tracker_name=tracker_name,
        tracker_params=tracker_params,
        output_dir=Path(args.output),
        dataset=args.dataset,
        resume=args.resume
    )
    evaluator.dump_per_frame = args.dump_per_frame

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
        max_videos=max_videos,
        visualize=args.visualize,
        subset=args.subset,
        repetitions=args.repetitions
    )


if __name__ == "__main__":
    main()
