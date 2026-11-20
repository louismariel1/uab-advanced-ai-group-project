# shared/validate_environment.py
"""Read-only validation and loading of Project A shared artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import torch
from torch.torch_version import TorchVersion
from torch import nn
import torch.nn.functional as F

EXPECTED_CLASSES = ["aquarium_fish", "flatfish", "ray", "shark", "trout"]
EXPECTED_BACKBONE_PARAMETERS = 495360

class ConvBackbone(nn.Module):
    """Exact Practical 3 backbone used in Notebook 1."""
    def __init__(self, num_features=64, input_channels=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.features = nn.Linear(128, num_features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return F.relu(self.features(x))

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_environment(project_root, require_ssl_checkpoint=True, device=None):
    """Validate Drive artifacts and return loaded teacher, config and metadata.

    Never downloads datasets, retrains, or modifies shared project files.
    Raises on failed mandatory checks.
    """
    root = Path(project_root)
    assert root.is_dir(), f"Project root missing: {root}"
    config_path = root / "config/project_config.json"
    split_path = root / "config/dataset_splits.json"
    ckpt_path = root / "checkpoints/ssl/teacher_ssl_v1.ckpt"
    for p in (config_path, split_path):
        assert p.is_file(), f"Required artifact missing: {p}"

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    assert cfg["seed"] == 0, "Unexpected seed"
    assert cfg["source_dataset"] == "CIFAR10", "Unexpected SSL source"
    assert cfg["target_dataset"] == "CIFAR100", "Unexpected target"
    assert cfg["target_classes"] == EXPECTED_CLASSES, "Target classes differ"
    assert cfg["target_train_fraction"] == 0.8, "Target split fraction differs"
    assert cfg["ssl_pretext"] == "rotation_prediction_4way", "SSL pretext differs"
    assert cfg["teacher_architecture"] == "ConvBackbone(num_features=64)", "Architecture differs"
    assert cfg["checkpoint_format"] == "state_dict_plus_metadata", "Checkpoint format differs"
    assert splits["seed"] == cfg["seed"], "Split seed differs"
    assert splits["classes"] == cfg["target_classes"], "Split classes differ"
    assert len(splits["original_class_ids"]) == 5, "Class ID count differs"
    assert len(set(splits["original_class_ids"])) == 5, "Duplicate class IDs"
    assert len(splits["train_indices"]) == 2000, "Train size differs"
    assert len(splits["val_indices"]) == 500, "Validation size differs"
    assert len(splits["test_indices"]) == 500, "Test size differs"
    for k, upper in (("train_indices", 50000), ("val_indices", 50000), ("test_indices", 10000)):
        ids = splits[k]
        assert len(set(ids)) == len(ids), f"Duplicate {k}"
        assert all(isinstance(i, int) and 0 <= i < upper for i in ids), f"Invalid {k}"
    assert not (set(splits["train_indices"]) & set(splits["val_indices"])), "Train/val overlap"
    assert splits["label_map"] == {
        str(class_id): index
        for index, class_id in enumerate(splits["original_class_ids"])
    }, "Target label map differs"

    result = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(split_path),
        "ssl_checkpoint_sha256": None,
        "ssl_checkpoint_status": "missing",
    }
    teacher = None
    payload = None
    if ckpt_path.is_file():
        checksum_path = ckpt_path.with_suffix(".sha256")
        assert checksum_path.is_file(), f"Checkpoint checksum missing: {checksum_path}"
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(ckpt_path)
        assert actual == expected, "SSL checkpoint SHA-256 mismatch"
        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert payload["config"]["target_classes"] == cfg["target_classes"], "Checkpoint target mismatch"
        assert payload["config"]["seed"] == cfg["seed"], "Checkpoint seed mismatch"
        assert payload["config"]["ssl_pretext"] == cfg["ssl_pretext"], "Checkpoint SSL pretext mismatch"
        assert payload["config"]["teacher_architecture"] == cfg["teacher_architecture"], "Checkpoint architecture mismatch"
        assert isinstance(payload.get("training_info"), dict), "Missing training metadata"
        assert payload["training_info"].get("pretext") == "rotation_prediction_4way", "Training pretext mismatch"
        assert payload["training_info"].get("epochs") == 30, "Incomplete SSL training"
        assert len(payload["training_info"].get("history", [])) == 30, "Training history incomplete"

        teacher = ConvBackbone(num_features=64)
        assert sum(p.numel() for p in teacher.parameters()) == EXPECTED_BACKBONE_PARAMETERS
        teacher.load_state_dict(payload["model_state_dict"], strict=True)
        selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        teacher = teacher.to(selected_device).eval()
        with torch.no_grad():
            sample = torch.zeros((2, 3, 32, 32), device=selected_device)
            output = teacher(sample)
        assert output.shape == (2, 64), f"Wrong teacher output: {tuple(output.shape)}"
        assert torch.isfinite(output).all().item(), "Nonfinite teacher output"
        result["ssl_checkpoint_sha256"] = actual
        result["ssl_checkpoint_status"] = "verified"
        result["teacher_output_shape"] = list(output.shape)
        result["teacher_parameters"] = EXPECTED_BACKBONE_PARAMETERS
        result["teacher_device"] = str(selected_device)
        result["best_ssl_val_accuracy"] = payload["training_info"].get("best_val_accuracy")
    elif require_ssl_checkpoint:
        raise FileNotFoundError(f"Frozen SSL teacher missing: {ckpt_path}")

    return {
        "teacher": teacher,
        "checkpoint_payload": payload,
        "config": cfg,
        "splits": splits,
        "report": result,
        "project_root": root,
    }
