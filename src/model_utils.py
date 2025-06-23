import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(model_dir: str, device: str = None):
    """
    Load the fine-tuned seq2seq model and tokenizer from disk.

    Args:
        model_dir (str): Directory path to the saved model.
        device (str, optional): Device to load the model onto ('cpu' or 'cuda').
                                Defaults to cuda if available.

    Returns:
        model: Loaded transformer model.
        tokenizer: Corresponding tokenizer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer, device


def extract_token_level_attention(input_text: str, model, tokenizer, device, max_length=1024):
    """
    Generates summary and extracts token-level attention weights from the model.

    Args:
        input_text (str): Clinical note text.
        model: Loaded seq2seq model.
        tokenizer: Corresponding tokenizer.
        device: Torch device.
        max_length (int): Max token length for input.

    Returns:
        summary (str): Generated summary text.
        attentions (dict): Token-level attention weights (decoder cross-attentions).
                           Format: List of attention tensors (layers x heads x tgt_len x src_len).
        input_tokens (list): List of input tokens.
        summary_tokens (list): List of generated summary tokens.
    """
    # Tokenize input and move to device
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    ).to(device)

    # Generate summary with attention outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            output_attentions=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )

    summary_ids = outputs.sequences
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Extract cross-attention weights (decoder cross attention to encoder)
    attentions = None
    if hasattr(outputs, "cross_attentions") and outputs.cross_attentions is not None:
        # attentions is a tuple: layers x (batch, heads, tgt_len, src_len)
        attentions = outputs.cross_attentions

    # Decode tokens for input and summary for visualization
    input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    summary_tokens = tokenizer.convert_ids_to_tokens(summary_ids[0])

    return summary, attentions, input_tokens, summary_tokens
