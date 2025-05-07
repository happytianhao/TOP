# 🚗💥 Nexar Dashcam Crash Prediction Challenge Solution: 1st Place on Public LB (0.926)

## 📄 Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [🥳 🚀 What's New](#--whats-new-)
- [📖 Introduction](#-introduction-)
- [🛠️ Installation](#️-installation-)
- [👨‍🏫 Get Started](#-get-started-)
- [🔋 Training](#-training-)
- [💡 Testing](#-testing-)
- [📊 Results](#-results-)
- [🎫 License](#-license-)
- [🖊️ Citation](#️-citation-)

## 🥳 🚀 What's New [🔝](#-table-of-contents)

- We achieve the **1st place** on the public leaderboard with score 0.926!!!

## 📖 Introduction [🔝](#-table-of-contents)
We independently trained three models separately:
- RGB Model: Take RGB frames as inputs.
- Flow Model: Take flow frames as inputs.
- Fuse Model: Take both RGB and flow frames as inputs to two different backbones with the same architecture and add their representations together for decoder head.

Each model is pre-trained on MM-AU (CAP-DATA, DADA-DATA) and D^2-City datasets and fine-tuned on the nexar dataset.

Moreover, we employed test-time flip augmentation and model ensembling, which improved the score from 0.889 (RGB) to 0.926 on the public leaderboard.

Each model follows an encoder-decoder achitecture. The encoder takes 5 frames as input within a 0.1s interval, then extract the feature with ResNet3dSlowOnly backbone.
The decoder donot predict the anomaly score directly but predict whether the accident will occur after 0.0s, 0.1s, 0.2s, ..., 2.0s. This will bring a more accurate supervision for the model which could help the model converge more stably. We output the maximum score across all the future timestamps 0.0s, 0.1s, 0.2s, ..., 2.0s to determine whether the accident will occur promptly.

See more details of our paper "Accident Anticipation via Temporal Occurrence Prediction" (Not available yet).

During training, we randomly sample 5 frames as input within a 0.1s interval before accident and the label correspond to the time to accident.
During testing, we sample 3 groups of 5 frames and the last frames of each group are the last, second last, third last frame of the test video, respectively. Then we average the output score of the 3 groups of input.

## 🛠️ Installation [🔝](#-table-of-contents)
```
conda create -n top python=3.8.5 -y
conda activate top
pip install torch torchvision
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.2.0
mim install mmaction2==1.2.0
git clone https://github.com/happytianhao/TOP.git -b nexar
cd TOP
pip install -v -e .
```

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
### RGB Model
To train the rgb model on the nexar dataset with pre-trained model, first modify the line 10 in `configs/predict_occurrence_snippet_nexar.py` from `modality = "fuse"` to `modality = "rgb"`, and then deannotate the line 272 to `load_from = "rgb_cdd_epoch_3_0.8567_0.9153.pth"` and annotate the line 276 to `# load_from = "fuse_n_epoch_13_0.9342_0.9298_0.905.pth"`.

### Flow Model
To train the flow model on the nexar dataset with pre-trained model, first modify the line 10 in `configs/predict_occurrence_snippet_nexar.py` from `modality = "fuse"` to `modality = "flow"`, and then deannotate the line 274 to `load_from = "flow_cdd_epoch_37_0.8026_0.8923.pth"` and annotate the line 276 to `# load_from = "fuse_n_epoch_13_0.9342_0.9298_0.905.pth"`.

### Fuse Model
No modification needs.

### Run
Run the following command for any above model:

Sigle-gpu:
```bash
python tools/train.py configs/predict_occurrence_snippet_nexar.py
```
Multi-gpu:
```bash
./dist_train.sh
```

## Testing [🔝](#-table-of-contents)
### RGB Model
To test the rgb model, first modify the line 10 in `configs/predict_occurrence_snippet_nexar.py` from `modality = "fuse"` to `modality = "rgb"`, and then run the following command:
```bash
python tools/test.py configs/predict_occurrence_snippet_nexar.py ckpts/rgb_n_epoch_50_0.9438_0.9163_0.883.pth
```

### Flow Model
To test the flow model, first modify the line 10 in `configs/predict_occurrence_snippet_nexar.py` from `modality = "fuse"` to `modality = "flow"`, and then run the following command:
```bash
python tools/test.py configs/predict_occurrence_snippet_nexar.py ckpts/flow_n_epoch_41_0.9094_0.9175_0.863.pth
```

### Fuse Model
To test the fuse model, run the following command:
```bash
python tools/test.py configs/predict_occurrence_snippet_nexar.py ckpts/fuse_n_epoch_13_0.9342_0.9298_0.905.pth
```

### Test Time Augmentation (Flip)
If you want to apply the flip test time augmentation, you should deannotate the line 164 in `configs/predict_occurrence_snippet_nexar.py` from `# dict(type="Flip", flip_ratio=1),` to `dict(type="Flip", flip_ratio=1),` and rerun the above commands.

You will get the submission file `outputs/sample_submission.csv`.

## Results [🔝](#-table-of-contents)

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
