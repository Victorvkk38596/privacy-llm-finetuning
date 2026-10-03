import torch
from transformers import AutoModelForSequenceClassification

def get_model(model_name: str, num_labels: int, train_layers: str):
    """
    Loads model and selectively freezes parameters.
    train_layers:
      - 'all': All parameters trainable
      - 'head': Only classification head trainable
      - '9,10,11': Only specified transformer layers + classification head
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    total_params = sum(p.numel() for p in model.parameters())

    if train_layers == "all":
        pass
    else:
        # 1. Freeze all backbone parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. Always unfreeze the classification head
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

        # 3. Unfreeze specific layer indices (e.g. "0,1,2" or "9,10,11")
        if train_layers != "head":
            selected_indices = [int(i.strip()) for i in train_layers.split(",")]
            for idx in selected_indices:
                layer_module = model.roberta.encoder.layer[idx]
                for param in layer_module.parameters():
                    param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fraction = (trainable_params / total_params) * 100

    print("=" * 60)
    print(f"Model: {model_name} | Configuration: {train_layers}")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({fraction:.2f}%)")
    print("=" * 60)

    return model, trainable_params, fraction
