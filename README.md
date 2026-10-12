# DeepGuard-AI

### Explainable and Robust Deepfake Detection

DeepGuard-AI is an AI research project investigating deepfake detection using deep learning, explainable AI (XAI), and robustness evaluation. The project aims to build a reproducible pipeline for distinguishing authentic facial images from manipulated content while examining model behavior, source-video leakage, and generalization.

**Repository:** https://github.com/zMHA/DeepGuard-AI

> **Project status:** Active research and development. The repository includes a ResNet-18 baseline, data validation, frame manifests, face cropping, crop auditing, group-based cross-validation, and automated tests. Current results are close to chance, so reliable deepfake detection has not yet been demonstrated.

---

## Table of Contents

* [Motivation](#motivation)
* [Research Objectives](#research-objectives)
* [Current Implementation](#current-implementation)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Dataset](#dataset)
* [Installation](#installation)
* [Workflow](#workflow)
* [Current Experimental Results](#current-experimental-results)
* [Evaluation Plan](#evaluation-plan)
* [Limitations](#limitations)
* [Roadmap](#roadmap)
* [Responsible Use](#responsible-use)
* [Citation](#citation)
* [License](#license)

## Motivation

Deepfake technology enables legitimate creative applications but can also facilitate impersonation, misinformation, and digital fraud. Detecting manipulated media remains challenging because manipulation methods evolve and machine-learning models can learn dataset-specific artifacts rather than generalizable evidence of manipulation.

DeepGuard-AI explores deep learning, explainability, and robustness evaluation to investigate these challenges. The long-term goal is to evaluate not only whether a model classifies an image correctly, but also how its predictions behave under changing input conditions and on previously unseen source videos.

## Research Objectives

1. **Deepfake detection:** Develop a binary-classification baseline for authentic and manipulated facial images.
2. **Explainable AI:** Investigate model predictions using visual explanation methods such as Grad-CAM.
3. **Robustness:** Evaluate performance under compression, blur, resizing, and other image transformations.
4. **Reproducibility:** Maintain structured manifests, experiment outputs, and automated tests.
5. **Leakage prevention:** Keep related source videos and their manipulated versions within the same evaluation group.
6. **Generalization:** Investigate performance on source videos and manipulation types not represented in training data.

## Current Implementation

| Component                                | Status                                          |
| ---------------------------------------- | ----------------------------------------------- |
| Manifest-based image dataset loader      | Implemented                                     |
| Dataset and image-path validation        | Implemented                                     |
| Video and source-group metadata          | Implemented; relationships require verification |
| Video-frame extraction pipeline          | Implemented                                     |
| Haar-cascade face cropping               | Implemented                                     |
| Face-crop manifest generation            | Implemented                                     |
| Face-crop auditing and failure reporting | Implemented                                     |
| ResNet-18 baseline training              | Implemented                                     |
| Model evaluation utilities               | Implemented                                     |
| Five-fold group cross-validation         | Implemented                                     |
| Automated tests                          | Implemented                                     |
| Grad-CAM visual explanations             | Planned                                         |
| Controlled robustness experiments        | Planned                                         |
| Interactive Streamlit demo               | Planned                                         |

Implementation status does not imply validated detection performance.

## Technology Stack

* **Language:** Python
* **Deep learning:** PyTorch and Torchvision
* **Baseline architecture:** ResNet-18
* **Computer vision:** OpenCV and Pillow
* **Data processing:** NumPy and Pandas
* **Evaluation:** scikit-learn
* **Visualization:** Matplotlib
* **Explainability:** Grad-CAM (planned)
* **Interactive application:** Streamlit (planned)
* **Testing:** pytest
* **Configuration:** YAML

The project has been developed in a Windows environment. CPU execution is possible, although training speed depends on the hardware and model configuration.

## Project Structure

```text
DeepGuard-AI/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                         # Local dataset; not tracked
│   └── processed/                   # Local frames, crops, and manifests
├── docs/
├── models/                          # Local model artifacts
├── notebooks/
├── results/                         # Local metrics and visualizations
├── scripts/
│   ├── audit_face_crops.py
│   ├── crop_faces.py
│   ├── cross_validate_face_crops.py
│   ├── download.py
│   ├── inspect_crop_failures.py
│   ├── inspect_frames.py
│   └── split_face_crops.py
├── src/
│   └── deepguard/
│       ├── __init__.py
│       ├── classifier.py
│       ├── dataset.py
│       ├── evaluate.py
│       ├── extract_frames.py
│       ├── prepare_data.py
│       ├── split_data.py
│       └── train.py
├── tests/
├── .gitignore
├── LICENSE
└── README.md
```

Local datasets, face crops, model checkpoints, and generated experiment outputs are excluded from version control.

## Dataset

### FaceForensics++

The initial experiments use [FaceForensics++](https://github.com/ondyari/FaceForensics), a research dataset containing authentic videos and manipulated facial videos.

The current local pilot contains 20 videos across five source groups. The processed face-crop dataset contains 97 images. This pilot supports pipeline development and preliminary evaluation only; it is not large enough to establish reliable real-world performance.

### Labels

| Label | Meaning            |
| ----: | ------------------ |
|   `0` | Real / authentic   |
|   `1` | Fake / manipulated |

### Manifest format

The processing pipeline uses CSV manifests containing:

| Column       | Description                                          |
| ------------ | ---------------------------------------------------- |
| `image_path` | Path to the image                                    |
| `label`      | Binary class label                                   |
| `video_id`   | Source video identifier                              |
| `group_id`   | Source-group identifier for leakage-aware evaluation |

Related original and manipulated videos should be assigned to the same source group whenever they share source material. The grouping logic must be verified against the dataset's actual relationships before final experiments.

**Important:** Frames or crops from the same source video must not be randomly distributed across training and validation sets. Such leakage can inflate evaluation results.

### Dataset access and terms

Obtain FaceForensics++ through its official access procedure and comply with its applicable terms of use. The dataset is not included in this repository.

* [Official repository](https://github.com/ondyari/FaceForensics)
* [FaceForensics++ paper](https://arxiv.org/abs/1901.08971)

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/zMHA/DeepGuard-AI.git
cd DeepGuard-AI
```

### 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install a compatible PyTorch and Torchvision combination using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Install the project utilities:

```powershell
python -m pip install numpy pandas scikit-learn opencv-python Pillow matplotlib PyYAML tqdm grad-cam streamlit pytest
```

Verify the environment:

```powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Workflow

Run commands from the repository root with the virtual environment activated. Use each script's `--help` option to inspect supported arguments.

### 1. Run automated tests

```powershell
python -m pytest -q
```

The latest recorded test run completed with six passing tests. Passing software tests verifies tested code paths, not model accuracy.

### 2. Prepare the frame manifest

Obtain the authorized dataset and prepare the frame manifest using the project's extraction pipeline and configuration. The expected manifest is:

`data/processed/manifest.csv`

The manifest should include `image_path`, `label`, `video_id`, and `group_id`.

### 3. Validate the dataset

```powershell
python -m src.deepguard.prepare_data --manifest data/processed/manifest.csv
```

If the command supports a separate image root, use the appropriate `--image-root` argument as documented by its help output.

### 4. Crop faces

```powershell
python scripts/crop_faces.py
```

The script uses OpenCV's Haar-cascade detector to generate face crops and a crop manifest. The default output locations are:

* Face crops: `data/processed/face_crops/`
* Crop manifest: `data/processed/face_crops_manifest.csv`
* Crop failures: `data/processed/crop_failures.csv`

Frames where no face is detected should be reviewed rather than silently assumed to contain valid crops.

### 5. Audit face crops

```powershell
python scripts/audit_face_crops.py
```

The audit reports class counts, source-group counts, missing image files, duplicate paths, and crop failures. A visual audit image is saved locally under `results/`.

### 6. Run group cross-validation

```powershell
python scripts/cross_validate_face_crops.py
```

The current cross-validation workflow evaluates the model across five source-group folds. Results are saved under `results/cv_face_crops/`.

Check the script's help output for supported options, including training epochs, batch size, learning rate, seed, and pretrained-weight settings.

### 7. Generate a fixed train/validation split

```powershell
python scripts/split_face_crops.py
```

**Caution:** The current split script assigns a specific source group to validation. It is tailored to the present pilot dataset and must be revised before use with different source groups or a new dataset.

## Current Experimental Results

The current local pilot contains 97 face crops from 20 videos across five source groups.

| Dataset audit                          | Result |
| -------------------------------------- | -----: |
| Real face crops                        |     48 |
| Fake face crops                        |     49 |
| Missing image files                    |      0 |
| Duplicate image paths                  |      0 |
| Frames with failed Haar face detection |      3 |

### Five-fold group cross-validation

| Metric               | Mean ± standard deviation |
| -------------------- | ------------------------: |
| Accuracy             |           0.5059 ± 0.0719 |
| Fake-class precision |           0.5146 ± 0.0525 |
| Fake-class recall    |           0.5733 ± 0.2127 |
| Fake-class F1-score  |           0.5254 ± 0.1109 |
| Validation loss      |           0.7191 ± 0.0345 |

These results are near chance level and vary across folds. The dataset is too small to support claims of reliable deepfake detection or generalization to unseen data. Improving data diversity, source-group validation, and model evaluation is the immediate priority.

The local dataset and generated experiment artifacts are excluded from Git, so reproducing these figures requires access to the same permitted source data and compatible processing setup.

## Evaluation Plan

### Classification metrics

Future experiments should report:

* Accuracy
* Precision and recall for each class
* F1-score
* ROC-AUC
* Confusion matrix

Evaluation should use source-group-disjoint data. Whenever enough data is available, a final held-out test set should remain untouched during model selection.

### Explainability

Grad-CAM visualizations will be used to investigate which image regions influence model predictions. Heatmaps are diagnostic aids, not proof that the model relies on meaningful or causal manipulation artifacts.

### Robustness

Planned experiments include:

* JPEG compression and recompression
* Gaussian blur
* Resizing and resampling
* Changes in image quality

Clean and transformed images should be evaluated using consistent samples and documented transformation settings.

### Generalization

Future work will investigate performance on unseen source videos, different manipulation methods, and different compression levels. Conclusions will be limited to the data and protocols actually evaluated.

## Limitations

* The pilot dataset contains only 97 face crops and five source groups.
* Current mean cross-validation accuracy is approximately 50.6%, indicating that the baseline is not reliable.
* Haar-cascade face detection misses some faces and can produce inconsistent crops.
* Source-group relationships require careful verification.
* The fixed validation-group script is specific to the current pilot.
* Grad-CAM explanations, robustness experiments, and the interactive demo remain planned work.
* Results from this small pilot cannot establish real-world detection performance.

No claims of state-of-the-art performance, production readiness, or reliable real-world detection are made.

## Roadmap

* [x] Implement manifest-based dataset loading and validation.
* [x] Implement the ResNet-18 baseline training pipeline.
* [x] Implement frame extraction and face-crop generation.
* [x] Add crop auditing and failure reporting.
* [x] Add source-group cross-validation.
* [x] Add automated tests for core pipeline behavior.
* [ ] Expand dataset diversity across source videos and manipulation methods.
* [ ] Verify source-group assignments against original dataset relationships.
* [ ] Improve face detection and cropping.
* [ ] Compare baseline architectures using the same leakage-aware protocol.
* [ ] Add Grad-CAM visual explanations.
* [ ] Conduct controlled robustness experiments.
* [ ] Evaluate on a suitably sized held-out dataset.
* [ ] Build an interactive Streamlit demonstration.
* [ ] Document reproducible experiments and findings.

## Responsible Use

DeepGuard-AI is intended for research, education, and media-forensics experimentation. Automated detection can produce false positives and false negatives, and performance may vary across datasets, compression settings, demographic groups, and manipulation methods.

Model predictions should not be treated as definitive evidence that a person created or distributed manipulated media. High-impact decisions require independent verification and appropriate human review.

Users are responsible for complying with dataset terms, privacy requirements, applicable law, and the rights of people depicted in media.

## Citation

If you use FaceForensics++ in research, consult the official repository and paper for the appropriate citation:

Rössler, A., Cozzolino, D., Verdoliva, L., Thies, J., and Nießner, M. “FaceForensics++: Learning to Detect Manipulated Facial Images.” *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019.

Paper: https://arxiv.org/abs/1901.08971

For DeepGuard-AI, cite the repository until a formal paper or archival release is available:

https://github.com/zMHA/DeepGuard-AI

## License

The source code is distributed under the license specified in the repository's `LICENSE` file, once that file is added and verified.

The software license does not grant rights to the FaceForensics++ dataset, third-party libraries, or other external materials. Those remain subject to their own terms and licenses.
