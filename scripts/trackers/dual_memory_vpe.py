"""
Dual Memory VPE System for YOLOe-VP-IoU Tracker

Implements a two-level memory system for Visual Position Embeddings (VPE):
- Long-term memory: Stable, high-quality VPE that represent object identity
- Short-term memory: Recent VPE with temporal decay for adaptation
- Anchor VPE: First VPE, permanently stored for baseline identity

Author: Roman Maslii
"""

from collections import deque
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np


class DualMemoryVPE:
    """
    Dual Memory Bank для VPE з балансом між довгостроковою стабільністю
    та короткостроковою адаптацією
    """

    def __init__(self,
                 long_term_capacity: int = 5,
                 long_term_quality_threshold: float = 0.7,
                 long_term_weight: float = 0.4,
                 short_term_capacity: int = 10,
                 short_term_weight: float = 0.6,
                 temporal_decay: float = 0.9,
                 update_long_term_every: int = 50,
                 replace_worst_lt: bool = True,
                 use_anchor: bool = True,
                 anchor_weight: float = 0.1,
                 verbose: bool = False):
        """
        Args:
            long_term_capacity: Максимальна кількість LT VPE
            long_term_quality_threshold: Мін. conf для додавання в LT
            long_term_weight: Вага LT в фінальній агрегації
            short_term_capacity: Максимальна кількість ST VPE (sliding window)
            short_term_weight: Вага ST в фінальній агрегації
            temporal_decay: Decay factor для старіших ST VPE (0-1)
            update_long_term_every: Оновлювати LT кожні N фреймів
            replace_worst_lt: Заміняти найгірший LT на кращий ST
            use_anchor: Використовувати anchor VPE в агрегації
            anchor_weight: Вага anchor VPE в фінальній агрегації
            verbose: Виводити debug інформацію
        """
        # Long-term memory (stable, rarely updated)
        self.long_term_vpe = deque(maxlen=long_term_capacity)
        self.long_term_conf = deque(maxlen=long_term_capacity)
        self.long_term_frames = deque(maxlen=long_term_capacity)

        # Short-term memory (dynamic, frequently updated)
        self.short_term_vpe = deque(maxlen=short_term_capacity)
        self.short_term_conf = deque(maxlen=short_term_capacity)
        self.short_term_age = deque(maxlen=short_term_capacity)

        # Anchor VPE (перший VPE, ніколи не видаляється)
        self.anchor_vpe = None
        self.anchor_conf = None
        self.anchor_frame = None

        # Configuration
        self.long_term_capacity = long_term_capacity
        self.long_term_quality_threshold = long_term_quality_threshold
        self.long_term_weight = long_term_weight
        self.short_term_capacity = short_term_capacity
        self.short_term_weight = short_term_weight
        self.temporal_decay = temporal_decay
        self.update_long_term_every = update_long_term_every
        self.replace_worst_lt = replace_worst_lt
        self.use_anchor = use_anchor
        self.anchor_weight = anchor_weight
        self.verbose = verbose

        # Stats
        self.frame_count = 0
        self.total_vpe_collected = 0
        self.lt_updates = 0

    def add_vpe(self, vpe: torch.Tensor, conf: float, frame_id: Optional[int] = None):
        """
        Додати новий VPE до dual memory

        Args:
            vpe: VPE tensor [1, D]
            conf: Detection confidence [0-1]
            frame_id: Номер фрейму (опційно)
        """
        if frame_id is None:
            frame_id = self.frame_count

        # 1. Зберегти anchor (перший VPE, ніколи не видаляється)
        if self.anchor_vpe is None:
            self.anchor_vpe = vpe.clone()
            self.anchor_conf = conf
            self.anchor_frame = frame_id
            if self.verbose:
                print(f"   🎯 Anchor VPE збережено (frame={frame_id}, conf={conf:.3f})")

        # 2. Завжди додаємо в short-term (sliding window)
        self.short_term_vpe.append(vpe.clone())
        self.short_term_conf.append(conf)
        self.short_term_age.append(0)  # Вік = 0 для нового VPE

        # Збільшуємо вік всіх існуючих ST VPE
        for i in range(len(self.short_term_age) - 1):
            self.short_term_age[i] += 1

        # 3. Перевірка для додавання в long-term
        should_add_to_lt = (
            conf >= self.long_term_quality_threshold and
            len(self.long_term_vpe) < self.long_term_capacity
        )

        if should_add_to_lt:
            self.long_term_vpe.append(vpe.clone())
            self.long_term_conf.append(conf)
            self.long_term_frames.append(frame_id)
            if self.verbose:
                print(f"   📌 LT VPE додано (#{len(self.long_term_vpe)}/{self.long_term_capacity}, conf={conf:.3f})")

        # 4. Періодичне оновлення long-term
        if self.frame_count > 0 and self.frame_count % self.update_long_term_every == 0:
            self._update_long_term_from_short_term()

        self.frame_count += 1
        self.total_vpe_collected += 1

    def _update_long_term_from_short_term(self):
        """Оновити LT memory найкращими ST VPE"""
        if len(self.short_term_vpe) == 0:
            return

        # Знайти найкращий ST VPE
        best_st_idx = np.argmax(list(self.short_term_conf))
        best_st_vpe = self.short_term_vpe[best_st_idx]
        best_st_conf = self.short_term_conf[best_st_idx]

        if len(self.long_term_vpe) < self.long_term_capacity:
            # Є місце - просто додати
            self.long_term_vpe.append(best_st_vpe.clone())
            self.long_term_conf.append(best_st_conf)
            self.long_term_frames.append(self.frame_count)
            self.lt_updates += 1
            if self.verbose:
                print(f"   🔄 LT VPE додано з ST (#{len(self.long_term_vpe)}/{self.long_term_capacity}, conf={best_st_conf:.3f})")
        elif self.replace_worst_lt:
            # Замінити найгірший LT якщо новий кращий
            worst_lt_idx = np.argmin(list(self.long_term_conf))
            worst_lt_conf = self.long_term_conf[worst_lt_idx]

            if best_st_conf > worst_lt_conf:
                self.long_term_vpe[worst_lt_idx] = best_st_vpe.clone()
                self.long_term_conf[worst_lt_idx] = best_st_conf
                self.long_term_frames[worst_lt_idx] = self.frame_count
                self.lt_updates += 1
                if self.verbose:
                    print(f"   🔄 LT VPE замінено (conf: {worst_lt_conf:.3f} → {best_st_conf:.3f})")

    def get_aggregated_vpe(self) -> Optional[torch.Tensor]:
        """
        Агрегація VPE з dual memory

        Returns:
            Агрегований VPE tensor [1, D] або None якщо немає VPE
        """
        # Якщо немає жодного VPE, повернути anchor або None
        if len(self.short_term_vpe) == 0 and len(self.long_term_vpe) == 0:
            return self.anchor_vpe if self.anchor_vpe is not None else None

        # 1. Агрегація Long-term VPE
        if len(self.long_term_vpe) > 0:
            lt_tensor = torch.cat(list(self.long_term_vpe), dim=0)
            lt_aggregated = lt_tensor.mean(dim=0, keepdim=True)
            lt_aggregated = F.normalize(lt_aggregated, p=2, dim=-1)
        else:
            lt_aggregated = None

        # 2. Агрегація Short-term VPE з temporal decay
        if len(self.short_term_vpe) > 0:
            st_weights = []
            for age in self.short_term_age:
                weight = self.temporal_decay ** age
                st_weights.append(weight)

            st_weights = torch.tensor(st_weights, dtype=torch.float32).unsqueeze(1)
            st_weights = st_weights / st_weights.sum()  # Normalize

            st_tensor = torch.cat(list(self.short_term_vpe), dim=0)

            # Ensure same device
            if st_tensor.device != st_weights.device:
                st_weights = st_weights.to(st_tensor.device)

            st_aggregated = (st_tensor * st_weights).sum(dim=0, keepdim=True)
            st_aggregated = F.normalize(st_aggregated, p=2, dim=-1)
        else:
            st_aggregated = None

        # 3. Комбінація LT та ST
        if lt_aggregated is not None and st_aggregated is not None:
            # Обидва доступні - зважена комбінація
            alpha = self.long_term_weight
            beta = self.short_term_weight
            total = alpha + beta

            final_vpe = (alpha/total) * lt_aggregated + (beta/total) * st_aggregated
            final_vpe = F.normalize(final_vpe, p=2, dim=-1)
        elif lt_aggregated is not None:
            final_vpe = lt_aggregated
        else:
            final_vpe = st_aggregated

        # 4. Додати anchor VPE з невеликою вагою
        if self.anchor_vpe is not None and self.use_anchor:
            anchor_w = self.anchor_weight
            final_vpe = (1 - anchor_w) * final_vpe + anchor_w * self.anchor_vpe
            final_vpe = F.normalize(final_vpe, p=2, dim=-1)

        return final_vpe

    def clear(self):
        """Очистити всю пам'ять (окрім anchor)"""
        self.long_term_vpe.clear()
        self.long_term_conf.clear()
        self.long_term_frames.clear()
        self.short_term_vpe.clear()
        self.short_term_conf.clear()
        self.short_term_age.clear()
        # Anchor залишається
        self.frame_count = 0
        self.total_vpe_collected = 0
        self.lt_updates = 0

    def reset(self):
        """Повний скид (включаючи anchor)"""
        self.clear()
        self.anchor_vpe = None
        self.anchor_conf = None
        self.anchor_frame = None

    def get_stats(self) -> Dict[str, Any]:
        """Отримати статистику dual memory"""
        return {
            'total_vpe_collected': self.total_vpe_collected,
            'anchor_available': self.anchor_vpe is not None,
            'anchor_conf': self.anchor_conf if self.anchor_vpe is not None else None,
            'long_term_count': len(self.long_term_vpe),
            'long_term_capacity': self.long_term_capacity,
            'long_term_avg_conf': np.mean(list(self.long_term_conf)) if len(self.long_term_conf) > 0 else 0.0,
            'short_term_count': len(self.short_term_vpe),
            'short_term_capacity': self.short_term_capacity,
            'short_term_avg_conf': np.mean(list(self.short_term_conf)) if len(self.short_term_conf) > 0 else 0.0,
            'lt_updates': self.lt_updates,
            'frame_count': self.frame_count,
        }

    def __len__(self) -> int:
        """Загальна кількість VPE (LT + ST, без anchor)"""
        return len(self.long_term_vpe) + len(self.short_term_vpe)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"DualMemoryVPE(anchor={'✓' if stats['anchor_available'] else '✗'}, "
                f"LT={stats['long_term_count']}/{stats['long_term_capacity']}, "
                f"ST={stats['short_term_count']}/{stats['short_term_capacity']}, "
                f"total={stats['total_vpe_collected']})")
