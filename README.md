# Advanced Machine Learning - Project A

## Efficient Image Classification

This group project investigates efficient image classification
using three complementary methods:

1. Self-Supervised Learning (SSL)
2. Low-Rank Adaptation (LoRA)
3. Knowledge Distillation (KD)

## Research Questions

### RQ1 - Teacher and Student Adaptation

How do teacher adaptation (Dense Fine-Tuning vs. LoRA),
knowledge distillation strategy (Output-Based KD vs.
conventional Feature-Based KD), and student adaptation
(Full-Rank vs. LoRA) affect student classification accuracy
and training efficiency when transferring knowledge from
an SSL-pretrained teacher?

RQ1 is assigned to Students 1 and 2.

### RQ2 - LoRA-Space Knowledge Distillation

Compare conventional Feature-Based KD with distillation
using LoRA intermediate representations, including Ax
and BAx.

RQ2 is assigned to Student 3.

## Shared Infrastructure

- CIFAR-10 rotation prediction for SSL pretraining
- CIFAR-100 five-class fish subset for downstream classification
- Fixed train, validation and test splits
- Shared SSL-pretrained teacher checkpoint
- Shared environment validation module

## Notebooks

| Notebook | Purpose |
|---|---|
| 01_Shared_Setup_Training.ipynb | SSL pretraining |
| 02_Environment_Validation.ipynb | Shared environment verification |
| 03_Student1_RQ1.ipynb | RQ1 pipelines P1-P3 |
| 03_Student2_RQ1.ipynb | RQ1 pipelines P4-P5 |
| 03_Student3_RQ2.ipynb | RQ2 experiments |

## Storage Policy

GitHub stores source code, configuration and documentation.

Google Drive stores large checkpoints, datasets and
generated experimental outputs.

The shared Google Drive project is located at:

MyDrive/Colab Notebooks/AI_Group_Project/

## Current Status

The shared SSL pretraining and environment validation
have been completed.

Full RQ1 and RQ2 experimental results are pending.
