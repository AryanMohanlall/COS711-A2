# COS711-A2
COS711 Assignment 2 - Segmenting solar filaments with deep learning
([Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026), scored with Panoptic Quality).

## Pipeline

The notebooks run in order; each one reads the files the earlier ones wrote (no notebook calls another).

| Notebook | What it does | Writes |
|---|---|---|
| `01_data_exploration` | Inventory, disk geometry, image quality, class imbalance, filament morphology, inter-annotator agreement | `artifacts/image_stats.csv`, `instance_stats.csv`, … |
| `02_cleaning_and_split` | Cleaning, preprocessing (limb correction + background flattening), label fusion, 5-fold time-grouped split | `artifacts/folds.csv`, `excluded_annotations.csv`, `cache/full_1024/*.npy` |
| `03_augmentation` | Augmentations calibrated to the measured variation between images | `artifacts/appearance_1024.csv` |
| `04_baseline_training` | U-Net (ResNet34 encoder), label comparison (each / soft / vote), post-processing tuning, error analysis, 5-fold cross-validation | `checkpoints/*.pt`, `artifacts/baseline_runs.csv` |
| `05_instance_separation` | Post-processing: hysteresis, gap grouping, boundary growth, test-time augmentation; checked on unseen folds | `artifacts/postprocess.json` |
| `06_compression` | Not done yet | |
| `07_inference_submission` | Calibration to the competition's ground truth, timed end-to-end test inference, `submission.csv` | `submission.csv`, `artifacts/inference.json` |

Shared code lives in `src/` (data loading, preprocessing, augmentation, dataset, model, training, post-processing, metrics).

## Results

| | PQ |
|---|---|
| Baseline, folds 1–4 (cross-validated) | 0.400 ± 0.009 |
| + gap grouping + TTA (notebook 05), folds 1–4 | 0.409 ± 0.013 |
| Final, calibrated to the competition's ground truth (notebook 07), folds 1–4 | 0.396 ± 0.011 |
| Kaggle public leaderboard | 0.34 |
| Two annotators against each other (for reference) | 0.36 |

The first two rows are scored against our OpenCV-rasterised masks; the third and the leaderboard against the
`pycocotools` masks the competition uses (about 10% smaller, so the same predictions score lower).

## Running

**Locally:** put the competition data under `data/` and install `requirements.txt`. Notebooks 01–03 run on CPU.

**On Kaggle** (04, 05, 07 need a GPU): turn on GPU and Internet; the first cell clones this repository for `src/`.
Attach as inputs:

| Notebook | Inputs |
|---|---|
| 04 | competition data, preprocessed images (`cache/full_1024` uploaded as a dataset) |
| 05 | + output of notebook 04 |
| 07 | + output of notebook 04 + output of notebook 05 |

`COS711_SMOKE=1` runs notebooks 04, 05 and 07 on a few images on CPU to check the pipeline.
