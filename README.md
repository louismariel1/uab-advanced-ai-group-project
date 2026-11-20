# Advanced Machine Learning — Project A

**Self-Supervised Learning, Low-Rank Adaptation and Knowledge Distillation**  
UAB · Advanced Machine Learning · Three-student group project

## Overview

This project studies whether knowledge from a self-supervised (SSL) teacher can be transferred to a smaller classifier efficiently, and how **LoRA** and **knowledge distillation (KD)** affect classification accuracy and training cost.

- **SSL source:** CIFAR-10, rotation prediction (four rotations), without using class labels for the pretext task.
- **Downstream target:** five CIFAR-100 classes: `aquarium_fish`, `flatfish`, `ray`, `shark`, `trout`.
- **Shared teacher:** `ConvBackbone(num_features=64)`, 495,360 parameters. Its recorded 59.86% rotation-validation accuracy is **not** downstream classification accuracy.
- **Target splits:** fixed 2,000 train / 500 validation / 500 test examples. Keep the provided split definitions unchanged.

**Current status:** shared SSL pretraining and environment initialization have been completed; the student notebooks currently contain validated setup and **experiment implementation placeholders**, not completed RQ1/RQ2 training pipelines. Do not interpret a successful initialization as completed experiments.

## Research questions and responsibilities

### RQ1 — Teacher adaptation, KD strategy and student adaptation

| Pipeline | Teacher adaptation | KD method | Student adaptation | Owner |
| --- | --- | --- | --- | --- |
| P1 (baseline) | Dense fine-tuning | Output KD | Full-rank | Student 1 |
| P2 | LoRA | Output KD | Full-rank | Student 1 |
| P3 | Dense fine-tuning | Feature KD | Full-rank | Student 1 |
| P4 | LoRA | Feature KD | Full-rank | Student 2 |
| P5 | LoRA | Output KD | LoRA | Student 2 |

Controlled comparisons: **P1–P2**, **P1–P3**, **P2–P4**, **P2–P5**. This is not a complete factorial design. Student 2 must reuse the **exact P2 LoRA-adapted teacher** created by Student 1 for P4 and P5.

### RQ2 — Conventional Feature KD versus LoRA-space KD

Student 3 compares conventional feature matching with distillation using the LoRA bottleneck `Ax` and adapter output `BAx` (with `Ax + BAx` as an optional variant). Student 3 uses the same frozen SSL checkpoint, configurations and splits, but does not depend on RQ1's student experiment outputs.

## Repository layout

```text
notebooks/
  01_Shared_Setup_Training.ipynb   # One-time shared SSL setup and pretraining
  02_Environment_Validation.ipynb # Standalone read-only environment check
  03_Student1_RQ1.ipynb           # Student 1: P1–P3
  03_Student2_RQ1.ipynb           # Student 2: P4–P5
  03_Student3_RQ2.ipynb           # Student 3: RQ2
shared/
  validate_environment.py        # Common validator and teacher loader
config/
  project_config.json            # Frozen shared settings
  dataset_splits.json            # Frozen target split indices
checkpoints/                      # Model files on Google Drive; not in Git
experiments/                      # Run artifacts on Google Drive
results/                          # Evaluation outputs on Google Drive
docs/                             # Tutorials and documentation
requirements.txt                  # Dependency manifest (if present)
```

GitHub stores the **code and small configuration files**. The shared **Google Drive** project holds the actual SSL checkpoint, training outputs and other large artifacts. A fresh clone alone is **not sufficient** to pass checkpoint validation.

## 1. Clone the repository

Install [Git](https://git-scm.com/downloads) if it is not already available, then run in a terminal (Windows PowerShell, WSL/Linux, or macOS):

```bash
git clone https://github.com/louismariel1/uab-advanced-ai-group-project.git
cd uab-advanced-ai-group-project
```

The repository is public for reading. Students who need to push must accept the GitHub collaborator invitation and authenticate with their own account. Do not commit secrets, private Drive credentials, checkpoints or large datasets.

## 2. Runtime and dependencies

**Recommended execution environment: Google Colab with Python 3 and a GPU runtime for training.** No C/C++ compilation or project build step is required: this is a Python/Jupyter research project.

Packages used in the shared setup notebook and validator:

| Dependency | Used for |
| --- | --- |
| Python 3 | Runtime |
| `torch` | Models, checkpoint loading, training and CUDA support |
| `torchvision` | CIFAR-10/CIFAR-100 and image transforms (`transforms.v2`) |
| `numpy` | Numerical operations and reproducibility |
| `scikit-learn` | Stratified dataset split |
| `Pillow` | Image handling |
| `google.colab` | Mounting Drive in the provided notebooks; available in Colab |

In Colab, check versions first; its environment commonly supplies these packages:

```python
import torch, torchvision, numpy, sklearn, PIL
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
```

If a dependency is missing in Colab, install it in a notebook cell and restart the runtime when prompted:

```python
%pip install numpy scikit-learn pillow
```

For PyTorch and torchvision, use **compatible versions** and follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for the chosen CPU/CUDA runtime; avoid replacing Colab's working CUDA stack unnecessarily. The source files inspected do **not** establish a pinned, tested dependency version matrix. If the repository's `requirements.txt` contains approved versions, use that manifest instead of guessing versions.

**Optional local environment** (for code editing or tests, not a drop-in replacement for the Drive-mounted Colab notebooks):

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS / WSL
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn pillow jupyter
# Install compatible torch and torchvision following pytorch.org
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The notebooks contain `google.colab` imports and fixed `/content/drive/...` paths, so **they will not run locally unchanged**. Use Colab for the supported end-to-end workflow.

## 3. Access the shared Google Drive project

The project owner shares the parent folder with each student as **Editor**:

```text
My Drive/Colab Notebooks/AI_Group_Project/
```

The Colab notebooks expect this exact path:

```python
PROJECT_ROOT = Path('/content/drive/MyDrive/Colab Notebooks/AI_Group_Project')
```

Required shared files include:

```text
config/project_config.json
config/dataset_splits.json
shared/validate_environment.py
checkpoints/ssl/teacher_ssl_v1.ckpt
checkpoints/ssl/teacher_ssl_v1.sha256
```

**Important:** GitHub and Google Drive are separate copies. The Drive folder must contain the matching notebook, validator and config versions; cloning or pulling from GitHub does not automatically update Drive. Coordinate changes to avoid divergent versions.

## 4. Run the project in Colab

1. Open **Google Drive → Colab Notebooks → AI_Group_Project → notebooks**.
2. Open your assigned notebook in Google Colab:
   - Student 1: `03_Student1_RQ1.ipynb`
   - Student 2: `03_Student2_RQ1.ipynb`
   - Student 3: `03_Student3_RQ2.ipynb`
3. Connect to a runtime (GPU recommended for training; CPU is sufficient for initialization).
4. Run **Cells 1–4** to mount Drive, import the common validator, verify configuration/splits/checkpoint, and load the frozen SSL teacher.
5. Confirm the validation status is `PASS`, checkpoint status is `verified`, and the teacher object was loaded.
6. Run the student-specific workspace initialization (Cell 5), then implement and execute the assigned training/evaluation cells as they are developed.

**Do not rerun Notebook 01**: shared SSL pretraining has already been performed. **Notebook 02 is optional** because each student notebook calls the same validator directly. At present, **Runtime → Run all** only validates and initializes the student notebook; it does **not** perform the unfinished experiments. Once training cells are added, `Run all` may start long-running training jobs.

### Optional standalone validation

Open `02_Environment_Validation.ipynb` in Colab and run its cells. It performs a read-only shared artifact audit; it does not download datasets, train models or modify checkpoints.

## 5. Outputs, evaluation and collaboration

Store generated files in the shared Drive project, using distinct student-specific paths under `experiments/`, `results/`, and `checkpoints/`.

- Student 1 publishes the **P2 LoRA teacher**, with its adapter configuration and required backbone weights, for Student 2.
- Record each run's seed, split IDs, preprocessing, hyperparameters, KD losses, teacher/student checkpoint references, training time and trainable parameters.
- Select models using **validation** data; use the held-out **test** set only for final evaluation.
- Compare downstream classification accuracy and training efficiency; where available, report peak memory, checkpoint size and inference cost.
- Never overwrite the frozen SSL teacher, shared config, split file or another student's outputs.

To sync source changes through GitHub (after your collaborator invitation has been accepted):

```bash
git pull --ff-only origin main
# Edit the relevant source files, then review changes:
git status
git diff
# Stage only intended source/config/docs files (not checkpoint outputs):
git add notebooks/03_Student1_RQ1.ipynb   # Example for Student 1; use your own path
git commit -m "Implement Student 1 RQ1 experiment updates"
git push origin main
```

Coordinate concurrent edits before pushing; consider a feature branch and pull request if more than one person is changing the same file.

## 6. Troubleshooting

| Symptom | Check |
| --- | --- |
| `PROJECT_ROOT` does not exist | Drive is mounted; folder is shared; path and capitalization match |
| `ModuleNotFoundError: shared` | `shared/validate_environment.py` exists in the **Drive** project root |
| Missing SSL checkpoint | Obtain access to the owner's Drive checkpoint; it is intentionally excluded from Git |
| Checkpoint checksum mismatch | Do not replace/retrain the frozen teacher; ask the owner to restore the validated file |
| CUDA unavailable | Enable **Runtime → Change runtime type → GPU**; CPU still works for setup checks |
| Push denied | Accept GitHub invitation and verify authentication and write permission |
| Notebook passes but produces no trained model | Training/evaluation pipelines have not yet been implemented in the starter notebooks |

## Completion criteria

The project is complete when the five RQ1 pipelines and the RQ2 variants have been implemented, executed, saved and evaluated reproducibly; all four RQ1 comparisons and the RQ2 comparison are supported by measured results. Environment-validation `PASS` alone does not satisfy these criteria.
