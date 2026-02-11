"""
Config Loader для трекер параметрів

Підтримує:
- YAML конфіги (.yaml, .yml)
- JSON конфіги (.json)
- Комбінування конфігів з параметрів командного рядка

Використання:
    from config_loader import ConfigLoader

    # Завантажити конфіг з файлу
    loader = ConfigLoader()
    params = loader.load('configs/yoloe-vp-iou.yaml')

    # Об'єднати з параметрами з командного рядка
    params = loader.merge(params, {'vpe_step': 10})
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ConfigMetadata:
    """Метадані конфігу"""
    name: str = ""
    description: str = ""
    tracker: str = ""
    created_date: str = ""
    version: str = "1.0"
    tags: list = None  # напр. ["fast_motion", "drone", "yoloe-vp-iou"]

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ConfigLoader:
    """Завантажувач конфігурацій для трекерів"""

    # Підтримувані розширення
    SUPPORTED_FORMATS = {'.yaml', '.yml', '.json'}

    # Директорія з конфігами за замовчуванням
    DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / 'configs'

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Args:
            config_dir: Директорія з конфігами (за замовч: scripts/../configs/)
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Завантажити конфіг з файлу

        Args:
            config_path: Шлях до файлу (абсолютний або відносно config_dir)

        Returns:
            Dict з параметрами трекера

        Raises:
            FileNotFoundError: Якщо файл не знайдено
            ValueError: Якщо формат файлу не підтримується
        """
        # Перетворити на Path
        path = Path(config_path)

        # Якщо відносний шлях - шукаємо в config_dir
        if not path.is_absolute():
            path = self.config_dir / path

        if not path.exists():
            raise FileNotFoundError(
                f"❌ Конфіг не знайдено: {path}\n"
                f"   Переконайтеся, що файл існує"
            )

        # Перевірити розширення
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"❌ Невідоме розширення файлу: {suffix}\n"
                f"   Підтримувані: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Завантажити залежно від типу
        if suffix in {'.yaml', '.yml'}:
            return self._load_yaml(path)
        else:  # .json
            return self._load_json(path)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Завантажити YAML конфіг"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if data is None:
                return {}

            # Витягти параметри (можуть бути під ключем 'params' або на верхньому рівні)
            if isinstance(data, dict):
                params = data.get('params', data)
                return self._extract_params(params)
            else:
                raise ValueError(f"Очікується dict, отримано {type(data)}")

        except yaml.YAMLError as e:
            raise ValueError(f"❌ Помилка парсингу YAML: {e}")
        except Exception as e:
            raise ValueError(f"❌ Помилка завантаження YAML: {e}")

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Завантажити JSON конфіг"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Витягти параметри (можуть бути під ключем 'params' або на верхньому рівні)
            if isinstance(data, dict):
                params = data.get('params', data)
                return self._extract_params(params)
            else:
                raise ValueError(f"Очікується dict, отримано {type(data)}")

        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Помилка парсингу JSON: {e}")
        except Exception as e:
            raise ValueError(f"❌ Помилка завантаження JSON: {e}")

    def _extract_params(self, data: Any) -> Dict[str, Any]:
        """
        Витягти параметри з data, ігноруючи метадані

        Метадані включають: name, description, created_date, version, tags
        Зберігає 'tracker' в параметрах, щоб можна було визначити тип трекера
        """
        if not isinstance(data, dict):
            return {}

        # 'tracker' залишаємо в параметрах для визначення типу трекера
        metadata_keys = {'name', 'description', 'created_date', 'version', 'tags', 'metadata', 'experiment_id'}

        params = {k: v for k, v in data.items() if k not in metadata_keys}

        return params

    def load_metadata(self, config_path: str) -> ConfigMetadata:
        """
        Завантажити метадані конфігу

        Args:
            config_path: Шлях до файлу

        Returns:
            ConfigMetadata
        """
        # Перетворити на Path
        path = Path(config_path)

        # Якщо відносний шлях - шукаємо в config_dir
        if not path.is_absolute():
            path = self.config_dir / path

        # Завантажити дані
        if path.suffix.lower() in {'.yaml', '.yml'}:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:  # .json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # Витягти метадані
        metadata_dict = {
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'tracker': data.get('tracker', ''),
            'created_date': data.get('created_date', ''),
            'version': data.get('version', '1.0'),
            'tags': data.get('tags', []),
        }

        return ConfigMetadata(**metadata_dict)

    def merge(self, base_params: Dict[str, Any],
              override_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Об'єднати два набори параметрів

        Args:
            base_params: Базові параметри (з файлу)
            override_params: Параметри для перевизначення (з командного рядка)

        Returns:
            Об'єднані параметри (override_params мають вищий пріоритет)
        """
        result = base_params.copy()
        result.update(override_params)
        return result

    def save(self, config_path: str, params: Dict[str, Any],
             metadata: Optional[ConfigMetadata] = None, format: str = 'yaml') -> Path:
        """
        Зберегти конфіг у файл

        Args:
            config_path: Шлях до файлу (абсолютний або відносно config_dir)
            params: Параметри трекера
            metadata: Опціональні метадані
            format: Формат ('yaml' або 'json')

        Returns:
            Path до збереженого файлу
        """
        # Перетворити на Path
        path = Path(config_path)

        # Якщо відносний шлях - шукаємо в config_dir
        if not path.is_absolute():
            path = self.config_dir / path

        # Додати розширення якщо потрібно
        if path.suffix == '':
            path = path.with_suffix('.yaml' if format == 'yaml' else '.json')

        # Підготувати дані
        data = params.copy()

        if metadata:
            # Додати метадані
            data['name'] = metadata.name
            data['description'] = metadata.description
            data['tracker'] = metadata.tracker
            data['created_date'] = metadata.created_date
            data['version'] = metadata.version
            data['tags'] = metadata.tags

        # Створити директорію якщо потрібно
        path.parent.mkdir(parents=True, exist_ok=True)

        # Зберегти
        try:
            if format == 'yaml' or path.suffix in {'.yaml', '.yml'}:
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:  # json
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            return path

        except Exception as e:
            raise ValueError(f"❌ Помилка збереження конфігу: {e}")

    def list_configs(self, pattern: Optional[str] = None) -> Dict[str, Path]:
        """
        Список всіх доступних конфігів

        Args:
            pattern: Опціональна маска для фільтрації (напр. "yoloe*", "*drone*")

        Returns:
            Dict з名а конфігу -> Path
        """
        if not self.config_dir.exists():
            return {}

        configs = {}

        for path in self.config_dir.rglob('*'):
            if path.is_file() and path.suffix in self.SUPPORTED_FORMATS:
                # Назва без расширення
                name = path.stem

                # Фільтрація по патерну
                if pattern:
                    from fnmatch import fnmatch
                    if not fnmatch(path.name, pattern):
                        continue

                configs[name] = path

        return dict(sorted(configs.items()))

    def validate(self, params: Dict[str, Any]) -> tuple[bool, list]:
        """
        Валідувати параметри

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Перевірити типи параметрів
        for key, value in params.items():
            # Перевірити спеціальні параметри
            if key == 'vpe_step' and not isinstance(value, int):
                errors.append(f"vpe_step має бути int, отримано {type(value)}")

            if key == 'max_vpe' and not isinstance(value, int):
                errors.append(f"max_vpe має бути int, отримано {type(value)}")

            if key in ['conf', 'iou_threshold', 'alpha_kf'] and not isinstance(value, (int, float)):
                errors.append(f"{key} має бути float, отримано {type(value)}")

        return len(errors) == 0, errors


def main():
    """CLI для управління конфігами"""
    import argparse

    parser = argparse.ArgumentParser(description="Менеджер конфігів для трекерів")

    subparsers = parser.add_subparsers(dest='command', help='Команда')

    # Команда: list
    list_parser = subparsers.add_parser('list', help='Список конфігів')
    list_parser.add_argument('--pattern', help='Маска для фільтрації')

    # Команда: load
    load_parser = subparsers.add_parser('load', help='Завантажити конфіг')
    load_parser.add_argument('config_path', help='Шлях до конфігу')

    # Команда: validate
    validate_parser = subparsers.add_parser('validate', help='Валідувати конфіг')
    validate_parser.add_argument('config_path', help='Шлях до конфігу')

    args = parser.parse_args()

    loader = ConfigLoader()

    if args.command == 'list':
        configs = loader.list_configs(args.pattern)
        if configs:
            print("\n📋 Доступні конфіги:")
            for name, path in configs.items():
                try:
                    metadata = loader.load_metadata(str(path))
                    desc = f" - {metadata.description}" if metadata.description else ""
                    print(f"  ✓ {name}{desc}")
                    if metadata.tags:
                        print(f"    Tags: {', '.join(metadata.tags)}")
                except:
                    print(f"  ✓ {name}")
        else:
            print("❌ Конфіги не знайдено")

    elif args.command == 'load':
        try:
            params = loader.load(args.config_path)
            print(f"\n✅ Конфіг завантажено:")
            print(json.dumps(params, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ Помилка: {e}")

    elif args.command == 'validate':
        try:
            params = loader.load(args.config_path)
            is_valid, errors = loader.validate(params)
            if is_valid:
                print("✅ Конфіг валідний")
            else:
                print("❌ Помилки валідації:")
                for error in errors:
                    print(f"  - {error}")
        except Exception as e:
            print(f"❌ Помилка: {e}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
