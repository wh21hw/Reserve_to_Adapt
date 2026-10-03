"""Dataset protocol shared by future baseline/IMP trainers; no training side effects.

Target labels are for protocol validation/evaluation only. Never feed them to an
unsupervised prototype estimator. VisDA requires an explicit original-id mapping.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass(frozen=True)
class TaskProtocol:
    name: str
    known_ids: tuple
    unknown_ids: tuple
    default_slots: int
    backbone_status: str

    def __post_init__(self):
        combined = self.known_ids + self.unknown_ids
        if not self.known_ids or not self.unknown_ids:
            raise ValueError("Both known and unknown sets must be nonempty")
        if len(set(combined)) != len(combined):
            raise ValueError("Duplicate/overlapping class IDs")
        if self.default_slots < 1:
            raise ValueError("Unknown capacity must be positive")

    @property
    def known_classes(self):
        return len(self.known_ids)

    def source_label(self, original_id):
        if original_id not in self.known_ids:
            raise ValueError(f"Source contains non-known class {original_id}")
        return self.known_ids.index(original_id)

    def evaluation_label(self, original_id):
        if original_id in self.known_ids:
            return self.source_label(original_id)
        if original_id in self.unknown_ids:
            return self.known_classes  # one semantic unknown, not a latent slot
        raise ValueError(f"Class {original_id} outside target protocol")


OFFICE31_A2W = TaskProtocol("office31-a2w", tuple(range(10)),
                           tuple(range(20, 31)), 2, "ResNet-50")
OFFICEHOME_PR2RW = TaskProtocol("officehome-pr2rw", tuple(range(25)),
                              tuple(range(25, 65)), 4, "ResNet-50")
VISDA_KNOWN_NAMES = ("bicycle", "bus", "car", "motorcycle", "train", "truck")


def macro_open_set_metrics(protocol, original_labels, semantic_predictions):
    """Preserve raw unknown classes for macro UNK, despite collapsed predictions.

    Predictions must be 0..C (one semantic unknown). This evaluation function
    deliberately accepts labels separately from unsupervised training inputs.
    Returns fractions, not percentages. All protocol classes must be present.
    """
    labels, predictions = list(original_labels), list(semantic_predictions)
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Nonempty labels and predictions must have equal length")
    counts = {label: [0, 0] for label in protocol.known_ids + protocol.unknown_ids}
    for label, prediction in zip(labels, predictions):
        expected = protocol.evaluation_label(label)
        if prediction not in range(protocol.known_classes + 1):
            raise ValueError("Expected collapsed semantic prediction, not latent slot")
        counts[label][0] += int(prediction == expected)
        counts[label][1] += 1
    if any(total == 0 for _, total in counts.values()):
        raise ValueError("Cannot report full-protocol macro metrics with absent classes")
    per_class = {label: correct / total for label, (correct, total) in counts.items()}
    known = sum(per_class[label] for label in protocol.known_ids) / protocol.known_classes
    unknown = sum(per_class[label] for label in protocol.unknown_ids) / len(protocol.unknown_ids)
    return {"OS_star": known, "UNK": unknown,
            "HOS": 2 * known * unknown / (known + unknown) if known + unknown else 0.0,
            "per_original_class": per_class}


def visda_protocol(class_id_to_name):
    """Use verified original IDs, never presume the six known IDs are 0..5."""
    if len(class_id_to_name) != 12:
        raise ValueError("VisDA requires all 12 classes")
    normalized = {key: name.strip().lower() for key, name in class_id_to_name.items()}
    if len(set(normalized.values())) != 12:
        raise ValueError("Class names must be unique")
    if not set(VISDA_KNOWN_NAMES).issubset(normalized.values()):
        raise ValueError("Missing a required VisDA known class")
    by_name = {name: key for key, name in normalized.items()}
    known = tuple(by_name[name] for name in VISDA_KNOWN_NAMES)
    unknown = tuple(sorted(set(normalized) - set(known)))
    return TaskProtocol("visda-synthetic2real", known, unknown, 2,
                        "UNRESOLVED: paper text ResNet-50, Table III VGGNet")


def read_list(list_path, image_root):
    """Parse paths with spaces; reject traversal, missing files and duplicates."""
    root = Path(image_root).resolve()
    rows, seen = [], set()
    for line_number, line in enumerate(Path(list_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            relative, label = line.rsplit(None, 1)
            label = int(label)
        except ValueError as error:
            raise ValueError(f"Malformed list row {line_number}") from error
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes root at row {line_number}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in seen:
            raise ValueError(f"Duplicate image at row {line_number}: {relative}")
        seen.add(path)
        rows.append((relative, label))
    if not rows:
        raise ValueError("Empty file list")
    return rows


def audit_lists(protocol, source_list, target_list, image_root):
    """Return an evaluation-side manifest; labels must stay out of IMP inputs."""
    source = read_list(source_list, image_root)
    target = read_list(target_list, image_root)
    for _, label in source:
        protocol.source_label(label)
    for _, label in target:
        protocol.evaluation_label(label)
    if {label for _, label in source} != set(protocol.known_ids):
        raise ValueError("Source is missing known classes")
    if {label for _, label in target} != set(protocol.known_ids + protocol.unknown_ids):
        raise ValueError("Target class set does not match the complete protocol")
    report = {"task": protocol.name, "known_classes": protocol.known_classes,
              "unknown_semantic_classes": len(protocol.unknown_ids),
              "baseline_unknown_slots": protocol.default_slots,
              "backbone_status": protocol.backbone_status,
              "evaluation_unknown_id": protocol.known_classes,
              "target_labels_for_training": False}
    for key, rows, path in (("source", source, source_list), ("target", target, target_list)):
        counts = {}
        for _, label in rows:
            counts[label] = counts.get(label, 0) + 1
        report[key] = {"samples": len(rows), "class_counts": counts,
                       "list_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    return report
