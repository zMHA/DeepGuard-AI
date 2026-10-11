# DeepGuard-AI

### Explainable and Robust Deepfake Detection

DeepGuard-AI is an AI research project focused on detecting manipulated facial videos using deep learning, explainable AI (XAI), and robustness evaluation. The project aims to develop a reproducible pipeline for distinguishing authentic media from manipulated content while investigating what visual evidence influences model predictions and how detection performance changes under common image degradations.

**Repository:** [github.com/zMHA/DeepGuard-AI](https://github.com/zMHA/DeepGuard-AI)

> **Project status:** Active development. The initial data-processing, model, training, and testing components are in place. Video-frame extraction, end-to-end evaluation, Grad-CAM explanations, robustness experiments, and the interactive demo are planned or under development.

---

## Table of Contents

* [Motivation](#motivation)
* [Research Objectives](#research-objectives)
* [Planned Features](#planned-features)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Dataset](#dataset)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Evaluation Plan](#evaluation-plan)
* [Research Methodology](#research-methodology)
* [Current Limitations](#current-limitations)
* [Roadmap](#roadmap)
* [Responsible Use](#responsible-use)
* [Citation](#citation)
* [License](#license)

## Motivation

Deepfake technology can be used for entertainment and legitimate creative applications, but manipulated media can also facilitate misinformation, impersonation, and digital fraud. Reliable detection remains challenging because manipulation methods evolve and models may learn dataset-specific patterns instead of meaningful manipulation artifacts.

DeepGuard-AI explores a combination of deep learning, visual explanations, and robustness testing to investigate these challenges.

The long-term goal is to develop a reproducible research pipeline that evaluates not only whether a model detects manipulated media, but also how it makes predictions and how consistently it performs under changing input conditions.

## Research Objectives

1. **Deepfake detection:** Develop a binary image-classification baseline for authentic and manipulated facial content.
2. **Explainable AI:** Use Grad-CAM to visualize image regions that contribute to model predictions.
3. **Robustness evaluation:** Investigate the effects of compression, blur, resizing, and other common image transformations.
4. **Reproducibility:** Maintain structured configurations, deterministic random seeds where practical, automated tests, and documented experiments.
5. **Leakage prevention:** Keep related source videos and their manipulated versions within the same dataset split.
6. **Practical deployment:** Develop an interactive demonstration for exploring predictions and visual explanations.

## Planned Features

| Feature                        | Technology           | Status                                                 |
| ------------------------------ | -------------------- | ------------------------------------------------------ |
| ResNet-18 binary classifier    | PyTorch, Torchvision | Baseline implementation                                |
| Manifest-based image dataset   | Pandas, Pillow       | Implemented                                            |
| Dataset validation             | Python               | Implemented                                            |
| Group-aware data splitting     | NumPy, Pandas        | Initial implementation; source-group validation needed |
| Training and checkpoint saving | PyTorch              | Implemented; real-data experiments pending             |
| Video-frame extraction         | OpenCV               | Planned                                                |
| Grad-CAM visual explanations   | Grad-CAM             | Planned                                                |
| Robustness experiments         | OpenCV, PyTorch      | Planned                                                |
| Interactive demo               | Streamlit            | Planned                                                |
| Automated tests                | Pytest               | Initial tests implemented                              |

These statuses describe the development state, not validated detection performance.

## Technology Stack

* **Language:** Python 3.12
* **Deep learning:** PyTorch and Torchvision
* **Baseline architecture:** ResNet-18
* **Computer vision:** OpenCV and Pillow
* **Data processing:** NumPy and Pandas
* **Machine learning utilities:** scikit-learn
* **Explainability:** Grad-CAM
* **Visualization:** Matplotlib
* **Interactive application:** Streamlit
* **Testing:** Pytest
* **Configuration:** YAML

Development and initial testing are performed on a CPU-based Windows environment. GPU acceleration may be introduced when compatible hardware is available.

## Project Structure

```text
DeepGuard-AI/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                  # Downloaded datasets (not tracked)
│   └── processed/            # Extracted frames and manifests (not tracked)
├── docs/                     # Research documentation
├── models/                   # Local model artifacts
├── notebooks/                # Experimental notebooks
├── results/                  # Evaluation outputs
├── scripts/
│   └── download.py           # FaceForensics++ downloader
├── src/
│   └── deepguard/
│       ├── __init__.py
│       ├── classifier.py     # ResNet-18 model
│       ├── dataset.py        # Manifest-based dataset loader
│       ├── extract_frames.py # Frame extraction (to be implemented)
│       ├── prepare_data.py   # Dataset validation
│       ├── split_data.py    # Dataset splitting
│       └── train.py         # Model training
├── tests/
│   ├── test_dataset.py
│   ├── test_split_data.py
│   └── test_training_smoke.py
├── .gitignore
└── README.md
```

## Dataset

### FaceForensics++

The initial experiments use [FaceForensics++](https://github.com/ondyari/FaceForensics), a research dataset containing authentic videos and manipulated facial videos.

The initial local pilot contains:

* 10 authentic videos.
* 10 Deepfakes-manipulated videos.
* Videos downloaded using the `c40` compression setting.
* Video files stored locally and excluded from the Git repository.

This small pilot is intended for pipeline development and debugging. It is not sufficient to establish reliable generalization or make broad claims about deepfake detection performance.

### Dataset labels

| Label | Meaning            |
| ----- | ------------------ |
| `0`   | Real / authentic   |
| `1`   | Fake / manipulated |

The image manifest is designed to include the following metadata:

| Column       | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| `image_path` | Path to an extracted image                                  |
| `label`      | Binary class label                                          |
| `video_id`   | Identifier for the video from which the image was extracted |
| `group_id`   | Shared source-group identifier used to prevent data leakage |

### Preventing data leakage

Multiple manipulated videos can be derived from the same source videos. If related samples appear in both training and validation sets, evaluation results may overestimate generalization.

DeepGuard-AI therefore aims to group related originals and manipulations before splitting the data. The grouping strategy must be verified against the dataset's source relationships before conducting final experiments.

**Important:** Frames extracted from the same video must never be treated as independent train/validation samples.

### Dataset access and terms

Obtain FaceForensics++ through its official access procedure and follow the dataset's applicable terms of use. The dataset is not included in this repository.

Official resources:

* [FaceForensics++ repository](https://github.com/ondyari/FaceForensics)
* [Official project page](https://fki.tic.heia-fr.ch/)
* [FaceForensics++ paper](https://arxiv.org/abs/1901.08971)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zMHA/DeepGuard-AI.git
cd DeepGuard-AI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current terminal, use:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install a PyTorch and Torchvision combination compatible with your operating system and hardware, following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Install the project utilities:

```powershell
python -m pip install numpy pandas scikit-learn opencv-python Pillow matplotlib PyYAML tqdm grad-cam streamlit pytest
```

Verify the environment:

```powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Configuration

The main configuration file is:

`configs/config.yaml`

The initial configuration includes:

| Parameter         | Initial value |
| ----------------- | ------------: |
| Random seed       |            42 |
| Image size        |     224 × 224 |
| Model             |     ResNet-18 |
| Number of classes |             2 |
| Batch size        |             8 |
| Training epochs   |             5 |
| Learning rate     |        0.0001 |
| Weight decay      |        0.0001 |
| Real label        |             0 |
| Fake label        |             1 |

These values are starting points for pipeline development, not optimized hyperparameters. Actual experiments may require different settings depending on dataset size, hardware, and evaluation results.

## Usage

Run commands from the repository root with the virtual environment activated.

### 1. Run the automated tests

```powershell
python -m pytest -v
```

The initial test suite covers dataset loading, label validation, data-splitting behavior, and a synthetic training smoke test.

Passing synthetic tests confirms basic software behavior; it does not demonstrate deepfake detection accuracy.

### 2. Validate a dataset manifest

The manifest validator expects a CSV with `image_path`, `label`, and `video_id` columns.

After preparing the manifest, run:

```powershell
python -m src.deepguard.prepare_data --manifest data/processed/manifest.csv
```

If the images are stored separately from the manifest's relative paths, provide the image root:

```powershell
python -m src.deepguard.prepare_data --manifest data/processed/manifest.csv --image-root data/processed
```

The validator checks required columns, labels, file availability, and class representation.

### 3. Prepare training and validation manifests

The splitting component is located at:

`src/deepguard/split_data.py`

Before using it for real experiments, verify that its grouping logic uses a common source-group identifier for each original video and all associated manipulated versions. The current initial implementation groups by `video_id`, which alone may not be sufficient to prevent source leakage.

### 4. Train the baseline model

Once frame extraction, manifest generation, source-group splitting, and validation are complete, the training script accepts separate training and validation manifests:

```powershell
python -m src.deepguard.train `
  --train-manifest data/processed/splits/train.csv `
  --validation-manifest data/processed/splits/validation.csv `
  --epochs 5 `
  --batch-size 8 `
  --learning-rate 0.0001
```

Optional arguments include `--image-root`, `--output-dir`, and `--seed`. Run the script with `--help` to inspect the available options.

The training script saves its best checkpoint under the configured output directory, which defaults to `results/checkpoints`.

Training on actual data should only begin after confirming that the extracted-image manifests match the dataset loader's expected paths and labels.

## Evaluation Plan

DeepGuard-AI aims to evaluate the following aspects.

### Classification performance

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

Metrics should be reported on a held-out, source-group-disjoint test set whenever sufficient data is available. Class-specific performance should be examined rather than relying on accuracy alone.

### Explainability

Grad-CAM visualizations will be used to investigate which image regions contribute to predictions. These visualizations are diagnostic aids, not proof that a model has learned causal or semantically correct manipulation features.

### Robustness

Planned perturbations include:

* JPEG compression and recompression.
* Gaussian blur.
* Image resizing and resampling.
* Changes in image quality.

Performance will be compared between clean and perturbed inputs using consistent evaluation samples and documented transformation settings.

### Generalization

Future experiments may compare performance across manipulation methods, compression levels, and unseen source videos. Conclusions will be limited to the datasets and protocols actually evaluated.

## Research Methodology

The intended pipeline is:

1. Obtain and document an authorized dataset.
2. Extract representative frames from authentic and manipulated videos.
3. Build a metadata manifest with labels, video identifiers, and source-group identifiers.
4. Validate the manifest and image files.
5. Split samples by source group to reduce leakage.
6. Train a ResNet-18 binary-classification baseline.
7. Evaluate classification performance on held-out data.
8. Generate Grad-CAM explanations.
9. Evaluate robustness under controlled perturbations.
10. Document results, limitations, and reproducibility settings.

## Current Limitations

* The available dataset pilot is small and intended for software validation.
* The frame-extraction component is not yet implemented.
* Source-group-aware splitting requires further validation before real-data experiments.
* Real-data performance metrics have not yet been established.
* Explainability and robustness experiments remain future work.
* The interactive Streamlit application is planned, not yet a completed feature.

No claims of state-of-the-art performance, production readiness, or reliable real-world detection are made at this stage.

## Roadmap

* [x] Set up the Python development environment.
* [x] Implement a ResNet-18 binary classifier.
* [x] Implement a manifest-based dataset loader.
* [x] Implement initial data validation and splitting components.
* [x] Implement the initial training pipeline and smoke tests.
* [x] Download a small FaceForensics++ pilot.
* [ ] Implement video-frame extraction.
* [ ] Generate image manifests with source-group metadata.
* [ ] Verify leakage-free train/validation/test splits.
* [ ] Train and evaluate the baseline on real data.
* [ ] Add Grad-CAM visualizations.
* [ ] Run controlled robustness experiments.
* [ ] Build an interactive Streamlit demonstration.
* [ ] Document reproducible experiments and findings.

## Responsible Use

DeepGuard-AI is intended for research, education, and media-forensics experimentation. Automated deepfake detection can produce false positives and false negatives, and its performance may vary across datasets, compression settings, demographic groups, and manipulation methods.

Model predictions should not be treated as definitive evidence that a person created or distributed manipulated media. High-impact decisions require independent verification and appropriate human review.

Users are responsible for complying with dataset licenses, privacy requirements, applicable law, and the rights of people depicted in media.

## Citation

If you use FaceForensics++ in your research, consult the official repository and paper for the appropriate citation:

Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., and Nießner, M. *FaceForensics++: Learning to Detect Manipulated Facial Images.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

Paper: https://arxiv.org/abs/1901.08971

For DeepGuard-AI itself, cite the repository URL until a formal paper or archival release is available:

`https://github.com/zMHA/DeepGuard-AI`

## License

No license has yet been specified for this repository. Unless and until a license is added, do not assume that the project's source code is licensed for unrestricted reuse.

The FaceForensics++ dataset is governed by its own terms and is not relicensed by this repository.
