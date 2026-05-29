"""
Train a spaCy textcat_multilabel classifier over the synthetic section
annotations and save it to ml/models/section_classifier/.

80/20 train/val split; per-label macro-accuracy reported. Target: ≥0.92 on the
held-out 20%. If accuracy is below target the script exits non-zero so CI
catches a regression.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import spacy
from spacy.training import Example

ML_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = ML_ROOT / "data" / "section_annotations" / "synthetic_sections.jsonl"
OUTPUT_DIR = ML_ROOT / "models" / "section_classifier"
LABELS = ["Education", "Experience", "Skills", "Summary", "Other"]
TARGET_ACCURACY = 0.92


def load_dataset() -> list[tuple[str, str]]:
    if not ANNOTATIONS.exists():
        sys.exit(f"Missing annotations at {ANNOTATIONS}. Run generate_section_training_data.py first.")
    pairs: list[tuple[str, str]] = []
    with open(ANNOTATIONS) as f:
        for line in f:
            row = json.loads(line)
            pairs.append((row["text"], row["label"]))
    return pairs


def split(pairs: list[tuple[str, str]], val_frac: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[str, str]]] = {}
    for text, label in pairs:
        by_label.setdefault(label, []).append((text, label))
    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def to_example(nlp, text: str, label: str) -> Example:
    cats = {lbl: 1.0 if lbl == label else 0.0 for lbl in LABELS}
    return Example.from_dict(nlp.make_doc(text), {"cats": cats})


def evaluate(nlp, val: list[tuple[str, str]]) -> tuple[float, dict[str, dict[str, int]]]:
    """Return (accuracy, per-label counts {tp, fn, fp, tn})."""
    counts = {lbl: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for lbl in LABELS}
    correct = 0
    for text, true_label in val:
        doc = nlp(text)
        pred = max(doc.cats.items(), key=lambda kv: kv[1])[0]
        if pred == true_label:
            correct += 1
        for lbl in LABELS:
            if lbl == true_label and lbl == pred:
                counts[lbl]["tp"] += 1
            elif lbl == true_label and lbl != pred:
                counts[lbl]["fn"] += 1
            elif lbl != true_label and lbl == pred:
                counts[lbl]["fp"] += 1
            else:
                counts[lbl]["tn"] += 1
    return correct / len(val), counts


def train(epochs: int = 20, seed: int = 42, dropout: float = 0.2) -> None:
    pairs = load_dataset()
    train_data, val_data = split(pairs)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat", last=True)
    for label in LABELS:
        textcat.add_label(label)

    train_examples = [to_example(nlp, t, l) for t, l in train_data]
    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    best_acc = 0.0
    patience, no_improve = 4, 0
    for epoch in range(epochs):
        random.seed(seed + epoch)
        random.shuffle(train_examples)
        losses: dict[str, float] = {}
        for example in train_examples:
            nlp.update([example], drop=dropout, sgd=optimizer, losses=losses)
        acc, counts = evaluate(nlp, val_data)
        print(f"Epoch {epoch+1:2d}: loss={losses.get('textcat', 0.0):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            nlp.to_disk(OUTPUT_DIR)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs (best={best_acc:.4f})")
            break

    print(f"\nFinal val_acc: {best_acc:.4f}")
    print("Per-label precision/recall/F1:")
    best_nlp = spacy.load(OUTPUT_DIR)
    _, counts = evaluate(best_nlp, val_data)
    for lbl in LABELS:
        c = counts[lbl]
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {lbl:12s} P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    if best_acc < TARGET_ACCURACY:
        print(f"\nAccuracy {best_acc:.3f} below target {TARGET_ACCURACY}", file=sys.stderr)
        sys.exit(1)
    print(f"\nModel saved to {OUTPUT_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train()
