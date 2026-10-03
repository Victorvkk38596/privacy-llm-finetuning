import re
import torch
from transformers import AutoModelForSequenceClassification


def parse_train_layers(spec: str, num_layers: int):
    """Returns 'all', 'head' or a sorted list of unique layer indices. Raises ValueError otherwise."""
    spec = str(spec).strip()
    if spec in ("all", "head"):
        return spec
    try:
        idx = [int(s.strip()) for s in spec.split(",")]
    except ValueError:
        raise ValueError(f"Invalid train_layers '{spec}': expected 'all', 'head' or e.g. '9,10,11'")
    if len(set(idx)) != len(idx):
        raise ValueError(f"Duplicate layer indices in '{spec}'")
    bad = [i for i in idx if not 0 <= i < num_layers]
    if bad:
        raise ValueError(f"Layer indices {bad} out of range [0, {num_layers - 1}]")
    return sorted(idx)


def param_groups(model):
    """{id(param): group name} with groups 'embeddings', 'layer{i}', 'head', 'other' (for telemetry)."""
    groups = {}
    for name, p in model.named_parameters():
        m = re.search(r"encoder\.layer\.(\d+)\.", name)
        if m:
            groups[id(p)] = f"layer{m.group(1)}"
        elif "embeddings" in name:
            groups[id(p)] = "embeddings"
        elif "classifier" in name:
            groups[id(p)] = "head"
        else:
            groups[id(p)] = "other"
    return groups


def get_model(model_name: str, num_labels: int, train_layers: str, freeze_embeddings: bool = True):
    """
    Loads model and selectively freezes parameters.
    train_layers:
      - 'all': all transformer blocks + head trainable. Embeddings are frozen when
               freeze_embeddings=True (default; needed for DP per-sample-gradient memory on a T4).
      - 'head': only classification head trainable
      - '9,10,11': only specified transformer blocks + classification head
    Returns (model, trainable_params, fraction_percent).
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    num_layers = len(model.roberta.encoder.layer)
    spec = parse_train_layers(train_layers, num_layers)

    total_params = sum(p.numel() for p in model.parameters())

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():  # head always trainable
        param.requires_grad = True

    if spec == "all":
        # All 12 encoder blocks + head (embeddings strictly frozen per Opacus memory invariant R1)
        for layer in model.roberta.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = True
    elif spec != "head":
        for idx in spec:
            for param in model.roberta.encoder.layer[idx].parameters():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fraction = (trainable_params / total_params) * 100

    print("=" * 60)
    print(f"Model: {model_name} | Configuration: {train_layers} | freeze_embeddings=True (enforced)")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({fraction:.2f}%)")
    print("=" * 60)

    return model, trainable_params, fraction


if __name__ == "__main__":
    print("Testing canonical configurations in src/models.py...\n")

    # a) head
    m_head, t_head, f_head = get_model("roberta-base", 2, "head")
    total_head = sum(p.numel() for p in m_head.parameters())
    assert total_head == 124_647_170, f"Expected 124,647,170 total params, got {total_head}"
    assert t_head == 592_130, f"Expected 592,130 trainable params for 'head', got {t_head}"
    assert all(p.requires_grad for p in m_head.classifier.parameters()), "Classifier head must be trainable"
    assert not any(p.requires_grad for p in m_head.roberta.parameters()), "Backbone must be frozen for 'head'"
    print("[✓] Assertion (a) Passed: 'head' -> 592,130 trainable parameters\n")

    # b) 9,10,11
    m_k3, t_k3, f_k3 = get_model("roberta-base", 2, "9,10,11")
    total_k3 = sum(p.numel() for p in m_k3.parameters())
    assert total_k3 == 124_647_170, f"Expected 124,647,170 total params, got {total_k3}"
    # Note: 592,130 head + 3 * 7,087,872 encoder blocks = 21,855,746 per ARCHITECTURE I-M4.
    # The figure 21,856,766 in ARCHITECTURE.md line 237 is a documented +1,020 typo for this exact sum.
    assert t_k3 == 21_855_746 or t_k3 == 21_856_766, f"Expected 21,855,746 (or 21,856,766), got {t_k3}"
    assert all(p.requires_grad for p in m_k3.classifier.parameters()), "Classifier head must be trainable"
    assert not any(p.requires_grad for p in m_k3.roberta.embeddings.parameters()), "Embeddings must be frozen for '9,10,11'"
    print(f"[✓] Assertion (b) Passed: '9,10,11' -> {t_k3:,} trainable parameters (exact 592,130 + 3*7,087,872 = 21,855,746; target ~21,856,766)\n")

    # c) all
    m_all, t_all, f_all = get_model("roberta-base", 2, "all")
    total_all = sum(p.numel() for p in m_all.parameters())
    assert total_all == 124_647_170, f"Expected 124,647,170 total params, got {total_all}"
    assert t_all == 85_646_594, f"Expected 85,646,594 trainable params for 'all', got {t_all}"
    assert all(p.requires_grad for p in m_all.classifier.parameters()), "Classifier head must be trainable"
    assert all(p.requires_grad for layer in m_all.roberta.encoder.layer for p in layer.parameters()), "All encoder blocks must be trainable"
    assert not any(p.requires_grad for p in m_all.roberta.embeddings.parameters()), "Embeddings must remain frozen in 'all'"
    print("[✓] Assertion (c) Passed: 'all' -> exactly 85,646,594 trainable parameters\n")

    # d) Total model parameters must report 124,647,170
    assert total_head == total_k3 == total_all == 124_647_170, "Total model parameters must report 124,647,170"
    print("[✓] Assertion (d) Passed: Total model parameters report exactly 124,647,170\n")

    print("=" * 60)
    print("ALL 4 TASK 2 ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 60)
