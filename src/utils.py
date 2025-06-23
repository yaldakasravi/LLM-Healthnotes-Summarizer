from typing import Dict
import torch

def preprocess_data(batch: Dict, tokenizer, config: Dict) -> Dict:
    """
    Tokenize and preprocess a batch of clinical notes and discharge summaries
    for seq2seq training.

    Args:
        batch: Dictionary with keys 'clinical_note' and 'discharge_summary', each
               a list of texts.
        tokenizer: Hugging Face tokenizer instance.
        config: Dictionary with config parameters such as max_input_length and max_output_length.

    Returns:
        Dictionary containing tokenized inputs and labels ready for model consumption.
    """

    inputs = batch["clinical_note"]
    targets = batch["discharge_summary"]

    # Tokenize clinical notes (inputs)
    model_inputs = tokenizer(
        inputs,
        max_length=config["max_input_length"],
        padding="max_length",
        truncation=True,
    )

    # Tokenize summaries (targets)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=config["max_output_length"],
            padding="max_length",
            truncation=True,
        )

    # Replace padding token id's of the labels by -100 so they are ignored by the loss function
    labels_input_ids = labels["input_ids"]
    labels_input_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels_input_ids
    ]

    model_inputs["labels"] = labels_input_ids

    return model_inputs


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across torch, numpy, random.

    Args:
        seed (int): The seed number.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
