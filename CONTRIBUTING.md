# Collaboration Guidelines

## Student Responsibilities

- Student 1: RQ1 pipelines P1, P2 and P3
- Student 2: RQ1 pipelines P4 and P5
- Student 3: RQ2 experiments

## Shared Infrastructure

Do not modify the frozen shared configuration, dataset
splits or SSL teacher checkpoint without team agreement.

## Development Workflow

1. Work in your assigned notebook.
2. Preserve the shared experimental configuration.
3. Record seeds, hyperparameters and evaluation metrics.
4. Save large checkpoints and experimental outputs to Drive.
5. Commit source-code and documentation changes to GitHub.
6. Use descriptive commit messages.

## Experimental Reproducibility

- Use the frozen dataset splits.
- Select models using validation data.
- Reserve test data for final evaluation.
- Report actual measurements; do not invent results.
- Distinguish training parameters from total parameters.

## Artifact Exchange

Student 1 provides the RQ1 LoRA teacher and P2
reference artifacts required by Student 2.

Student 2 provides the P4 and P5 results for the
joint RQ1 comparison.

Student 3 works independently from the shared
SSL-pretrained teacher and common infrastructure.
