# TOP: Temporal Occurrence Prediction for Accident Anticipation

Official implementation of **"Accident Anticipation via Temporal Occurrence Prediction"**.

📄 [**Paper (arXiv)**](https://arxiv.org/abs/2510.22260) | 🌐 [**Project Page**](https://happytianhao.github.io/TOP/)

## Overview

This repository contains the code for our traffic accident anticipation framework based on **Temporal Occurrence Prediction (TOP)**. Our method predicts potential collisions in an online manner by modeling the temporal occurrence of accidents, enabling timely alerts to enhance road safety.

### Key Features
- **Temporal Occurrence Prediction**: Models when accidents will occur through temporal prediction
- **Online Prediction**: Real-time accident anticipation for practical deployment
- **Multi-dataset Support**: Unified framework supporting CAP, DADA, and Nexar datasets
- **Strong Performance**: Achieves state-of-the-art results on benchmark datasets

## Installation

### Prerequisites
- Python 3.8.5
- PyTorch 1.3+
- CUDA (recommended)

### Installation Steps

```bash
# Step 1: Create Environment
conda create -n top python=3.8.5 -y
conda activate top

# Step 2: Install PyTorch
pip install torch torchvision

# Step 3: Install MMAction2 Dependencies
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.2.0

# Step 4: Clone and Install TOP
git clone https://github.com/happytianhao/TOP.git
cd TOP
pip install -v -e .
```

## Data Preparation

### Dataset Structure

The project expects datasets to be organized as follows:

```
TOP/
├── data/
│   ├── MM-AU/                    # Symlink to actual dataset location
│   │   ├── CAP-DATA/
│   │   │   ├── 1-10/            # Video frame directories grouped by range
│   │   │   │   ├── 1/           # Video ID
│   │   │   │   │   ├── 001537/  # Scene ID
│   │   │   │   │   │   ├── images/
│   │   │   │   │   │   │   ├── 000001.jpg
│   │   │   │   │   │   │   ├── 000002.jpg
│   │   │   │   │   │   │   └── ...
│   │   │   │   │   │   └── labels/
│   │   │   │   │   ├── 002004/
│   │   │   │   │   └── ...
│   │   │   │   ├── 2/
│   │   │   │   └── ...
│   │   │   ├── 11/
│   │   │   ├── 12-42/
│   │   │   ├── 43/
│   │   │   ├── 44-62/
│   │   │   ├── cap_train_annotations.csv
│   │   │   ├── cap_val_annotations.csv
│   │   │   └── cap_val_references.csv
│   │   └── DADA-DATA/
│   │       ├── 1/               # Video ID
│   │       │   ├── 001/         # Scene ID
│   │       │   │   ├── images/
│   │       │   │   │   ├── 0001.png
│   │       │   │   │   ├── 0002.png
│   │       │   │   │   └── ...
│   │       │   │   └── labels/
│   │       │   ├── 002/
│   │       │   └── ...
│   │       ├── 2/
│   │       ├── ...
│   │       ├── 61/
│   │       ├── dada_train_annotations.csv
│   │       ├── dada_val_annotations.csv
│   │       └── dada_val_references.csv
│   └── nexar-collision-prediction/  # Symlink to Nexar dataset
│       ├── train/               # Training videos
│       │   ├── 00000.mp4
│       │   ├── 00003.mp4
│       │   └── ...
│       ├── test/                # Test videos
│       ├── train_raw_frames/    # Extracted training frames
│       │   ├── 00000/
│       │   │   ├── 000000.jpg
│       │   │   ├── 000001.jpg
│       │   │   └── ...
│       │   └── ...
│       ├── test_raw_frames/     # Extracted test frames
│       ├── nexar_train_annotations.csv
│       ├── nexar_val_annotations.csv
│       └── nexar_val_references.csv
```

### Download Datasets

1. **CAP & DADA Datasets**: 
   - Baidu Netdisk: [https://pan.baidu.com/s/1HrA8BibdpgcGiS6lQyDF8A?pwd=mskd](https://pan.baidu.com/s/1HrA8BibdpgcGiS6lQyDF8A?pwd=mskd)
   - Hugging Face: [https://huggingface.co/datasets/JeffreyChou/MM-AU](https://huggingface.co/datasets/JeffreyChou/MM-AU/tree/main)
2. **Nexar Dataset**: [Kaggle Competition](https://www.kaggle.com/competitions/nexar-collision-prediction/data)

### Download Annotation Files

The preprocessed annotation CSV files are available at:
- 📦 [**Google Drive**](https://drive.google.com/drive/folders/16Z2VePTQnxFuyQsaKcfJTUtiRCwwg1Eq?usp=drive_link)

Download the following files and place them in the corresponding dataset directories:
- `cap_train_annotations.csv`, `cap_val_annotations.csv`, `cap_val_references.csv` → `data/MM-AU/CAP-DATA/`
- `dada_train_annotations.csv`, `dada_val_annotations.csv`, `dada_val_references.csv` → `data/MM-AU/DADA-DATA/`
- `nexar_train_annotations.csv`, `nexar_val_annotations.csv`, `nexar_val_references.csv` → `data/nexar-collision-prediction/`

### Video Frame Extraction

**Note**: CAP and DADA datasets already contain extracted frames. Only Nexar dataset requires frame extraction.

#### Extract Nexar Frames

```bash
# Make sure you have opencv-python installed
pip install opencv-python

# Extract frames from Nexar videos
python extract_frames.py
```

The script will extract frames from:
- `data/nexar-collision-prediction/train/*.mp4` → `data/nexar-collision-prediction/train_raw_frames/`
- `data/nexar-collision-prediction/test/*.mp4` → `data/nexar-collision-prediction/test_raw_frames/`

## Training

### Train on CAP Dataset
```bash
# Single GPU
python tools/train.py configs/accident_anticipation/top/top_cap.py

# Multiple GPUs (e.g., 4 GPUs)
bash tools/dist_train.sh configs/accident_anticipation/top/top_cap.py 4
```

### Train on DADA Dataset
```bash
# Single GPU
python tools/train.py configs/accident_anticipation/top/top_dada.py

# Multiple GPUs
bash tools/dist_train.sh configs/accident_anticipation/top/top_dada.py 4
```

### Train on Nexar Dataset
```bash
# Single GPU
python tools/train.py configs/accident_anticipation/top/top_nexar.py

# Multiple GPUs
bash tools/dist_train.sh configs/accident_anticipation/top/top_nexar.py 4
```

### Training Configuration

Key hyperparameters can be adjusted in the config files:
- `clip_len`: Number of frames per clip (default: 5)
- `num_clips`: Number of clips to sample (default: 30)
- `batch_size`: Batch size per GPU (default: 2)
- `max_epochs`: Training epochs (default: 50)

## Evaluation

### Test on CAP Dataset
```bash
# Single GPU
python tools/test.py configs/accident_anticipation/top/top_cap.py work_dirs/top_cap/best_mAUC_epoch_XX.pth

# Multiple GPUs
bash tools/dist_test.sh configs/accident_anticipation/top/top_cap.py work_dirs/top_cap/best_mAUC_epoch_XX.pth 4
```

### Test on DADA Dataset
```bash
python tools/test.py configs/accident_anticipation/top/top_dada.py work_dirs/top_dada/best_mAUC_epoch_XX.pth
```

### Test on Nexar Dataset
```bash
python tools/test.py configs/accident_anticipation/top/top_nexar.py work_dirs/top_nexar/best_mAUC_epoch_XX.pth
```

### Evaluation Metrics

The framework reports the following metrics:
- **mAP**: Mean Average Precision (average of AP@0.5s, AP@1.0s, AP@1.5s)
- **mAUC**: Mean Area Under ROC Curve (average of AUC@0.5s, AUC@1.0s, AUC@1.5s, computed at FPR ≤ 10%)
- **mAUC^0.1**: Mean AUC at FPR ≤ 10% for low false alarm scenarios
- **TTA**: Time-to-Accident, measuring how early the model detects accidents
- **TTA^0.1**: Time-to-Accident at FAR ≤ 10%
- **AP@t**: Average Precision at t seconds (0.0s, 0.5s, 1.0s, 1.5s) before accident
- **AUC@t**: Area Under ROC Curve at t seconds before accident

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{zhao2025accident,
  title={Accident Anticipation via Temporal Occurrence Prediction},
  author={Zhao, Tianhao and Zou, Yiyang and Mao, Zihao and Xiao, Peilun and Huang, Yulin and Yang, Hongda and Li, Yuxuan and Li, Qun and Wu, Guobin and Lin, Yutian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

### Follow-up Work

Check out our follow-up work on collision-anchored risk propagation:

**RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation**  
*Yiyang Zou, Tianhao Zhao, Peilun Xiao, Hongyu Jin, Longyu Qi, Yuxuan Li, Liyin Liang, Yifeng Qian, Chunbo Lai, Yutian Lin, Zhihui Li, Yu Wu*  
CVPR 2026 Highlight  
📄 [Paper](https://arxiv.org/abs/2603.27165)

```bibtex
@inproceedings{zou2026riskprop,
  title={RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation},
  author={Zou, Yiyang and Zhao, Tianhao and Xiao, Peilun and Jin, Hongyu and Qi, Longyu and Li, Yuxuan and Liang, Liyin and Qian, Yifeng and Lai, Chunbo and Lin, Yutian and Li, Zhihui and Wu, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This codebase is built upon [MMAction2](https://github.com/open-mmlab/mmaction2). We thank the OpenMMLab team for their excellent work.

## Contact

For questions or issues, please open an issue or contact: zthwhucs@gmail.com
