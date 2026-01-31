# Installation Guide for YOLOe-VP-IoU Tracker

## Prerequisites
- Python >= 3.8
- CUDA 11.8+ (for GPU support)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/yoloe_tracker.git
cd yoloe_tracker
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

#### Basic installation (for tracking only)
```bash
pip install -r requirements.txt
```

#### Development installation (with testing and code quality tools)
```bash
pip install -r requirements-dev.txt
```

### 4. Install the tracker as a package (optional)
```bash
pip install -e .
```

## Verify Installation

Run the quick test to verify everything is working:
```bash
python -c "from scripts.trackers.yoloe_vp_iou_tracker import YOLOeVPIoUTracker; print('✅ Installation successful!')"
```

## Troubleshooting

### CUDA/GPU Issues
If you have GPU errors, you may need to install the correct PyTorch version for your CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Missing Models
Models will be automatically downloaded on first use. Ensure you have internet connectivity and sufficient disk space (~2-3 GB).

## Configuration

Before running tracking, ensure your configuration files are properly set:
- YOLOe model: `configs/yoloe-v8s-seg.pt`
- Tracker config: `configs/yoloe-vp-iou/drone-advanced.yaml`

## Next Steps

- Read the [README.md](README.md) for usage examples
- Check [configs/](configs/) for available configurations
- See [scripts/modular_evaluation.py](scripts/modular_evaluation.py) for evaluation examples
