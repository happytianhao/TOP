# 🚗💥 Nexar Dashcam Crash Prediction Challenge Solution: 1st Place on Public LB (0.926)

## 📄 Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [🥳 🚀 What's New](#--whats-new-)
- [📖 Introduction](#-introduction-)
- [🎁 Major Features](#-major-features-)
- [🛠️ Installation](#️-installation-)
- [👀 Model Zoo](#-model-zoo-)
- [👨‍🏫 Get Started](#-get-started-)
- [🎫 License](#-license-)
- [🖊️ Citation](#️-citation-)

## 🥳 🚀 What's New [🔝](#-table-of-contents)


## 📖 Introduction [🔝](#-table-of-contents)

MMAction2 is an open-source toolbox for video understanding based on PyTorch.
It is a part of the [OpenMMLab](http://openmmlab.com/) project.

<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_overview.gif" width="380px">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px">
  <p style="font-size:1.5vw;"> Action Recognition on Kinetics-400 (left) and Skeleton-based Action Recognition on NTU-RGB+D-120 (right)</p>
</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="580px"/><br>
    <p style="font-size:1.5vw;">Skeleton-based Spatio-Temporal Action Detection and Action Recognition Results on Kinetics-400</p>
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/spatio-temporal-det.gif" width="800px"/><br>
    <p style="font-size:1.5vw;">Spatio-Temporal Action Detection Results on AVA-2.1</p>
</div>

## 🎁 Major Features [🔝](#-table-of-contents)

- **Modular design**: We decompose a video understanding framework into different components. One can easily construct a customized video understanding framework by combining different modules.

- **Support five major video understanding tasks**: MMAction2 implements various algorithms for multiple video understanding tasks, including action recognition, action localization, spatio-temporal action detection, skeleton-based action detection and video retrieval.

- **Well tested and documented**: We provide detailed documentation and API reference, as well as unit tests.

## 🛠️ Installation [🔝](#-table-of-contents)

MMAction2 depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine), [MMDetection](https://github.com/open-mmlab/mmdetection) (optional) and [MMPose](https://github.com/open-mmlab/mmpose) (optional).

Please refer to [install.md](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) for detailed instructions.

<details close>
<summary>Quick instructions</summary>

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

</details>


## 👨‍🏫 Get Started [🔝](#-table-of-contents)
You can download all the checkpoints and annotations on [Google Drive](https://drive.google.com/drive/folders/1OyYOKteAsNUcQsb3wqi2_8wwOgXdK7e7?usp=drive_link), and then arrange the folders and files like:

```
.
├── ckpts
│   ├── rgb_cdd_epoch_3_0.8567_0.9153.pth
│   ├── rgb_n_epoch_50_0.9438_0.9163_0.883.pth
│   ├── flow_cdd_epoch_37_0.8026_0.8923.pth
│   ├── flow_n_epoch_41_0.9094_0.9175_0.863.pth
│   └── fuse_n_epoch_13_0.9342_0.9298_0.905.pth
├── data
│   ├── nexar-collision-prediction
│   │   ├── train
│   │   │   ├── 00000.mp4
│   │   │   ├── 00003.mp4
│   │   │   └── ...
│   │   ├── test
│   │   │   ├── 00001.mp4
│   │   │   ├── 00002.mp4
│   │   │   └── ...
│   │   └── annotations.csv
│   └── ...
└── ...
```

## Training [🔝](#-table-of-contents)

## Testing [🔝](#-table-of-contents)
### RGB Model
To test the pre-trained rgb model, first modify the line 10 in `configs/predict_occurrence_snippet_nexar.py` from `modality = "fuse"` to `modality = "rgb"`, and then run the following command:
```bash
python tools/test.py configs/predict_occurrence_snippet_nexar.py ckpts/rgb_n_epoch_50_0.9438_0.9163_0.883.pth
```

### Flow Model
To test the pre-trained flow model, first modify the line 10 in `configs/predict_occurrence_snippet_nexar.py` from `modality = "fuse"` to `modality = "flow"`, and then run the following command:
```bash
python tools/test.py configs/predict_occurrence_snippet_nexar.py ckpts/flow_n_epoch_41_0.9094_0.9175_0.863.pth
```

### Fuse Model
To test the pre-trained fuse model, run the following command:
```bash
python tools/test.py configs/predict_occurrence_snippet_nexar.py ckpts/fuse_n_epoch_13_0.9342_0.9298_0.905.pth
```

### Test Time Augmentation (Flip)
If you want to apply the flip test time augmentation, you should deannotate the line 164 in `configs/predict_occurrence_snippet_nexar.py` from `# dict(type="Flip", flip_ratio=1),` to `dict(type="Flip", flip_ratio=1),` and rerun the above commands.

You will get the submission file `outputs/sample_submission.csv`.

## Results (Public LB) [🔝](#-table-of-contents)

|Exp.|Model / Ensemble|Score (Public LB)|
|---|---|---|
|1|RGB|0.889|
|2|RGB w/ flip|0.870|
|3|Flow|0.867|
|4|Flow w/ flip|0.889|
|5|Fuse|0.907|
|6|Fuse w/ flip|0.883|
|7|RGB (0.7 \* Exp. 1 + 0.3 \* Exp. 2)|0.892|
|8|Flow (0.3 \* Exp. 3 + 0.7 \* Exp. 4)|0.892|
|9|RGB + Flow (0.5 \* Exp. 7 + 0.5 \* Exp. 8)|0.923|
|10|RGB + Flow + Fuse (0.8 \* Exp. 9 + 0.2 \* Exp. 5)|**0.926**|

## 🎫 License [🔝](#-table-of-contents)

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation [🔝](#-table-of-contents)

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```
