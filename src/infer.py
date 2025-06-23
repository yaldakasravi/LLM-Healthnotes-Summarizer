import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def generate_summary(input_text, model, tokenizer, device, max_length=256):
    """
    Generate a summary for a single clinical note.

    Args:
        input_text (str): The clinical note to summarize.
        model: Hugging Face Seq2Seq model.
        tokenizer: Corresponding tokenizer.
        device: torch device (cpu or cuda).
        max_length (int): Maximum length for generated summary.

    Returns:
        summary (str): Generated summary text.
        attentions (list): Optional attention weights (currently None).
    """
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length",
    ).to(device)

    # Enable model eval mode and no_grad for inference
    model.eval()
    with torch.no_grad():
        # Generate summary ids and optionally return attentions
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            output_attentions=True,      # to get attentions (if supported)
            return_dict_in_generate=True
        )

    summary_ids = outputs.sequences
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Extract decoder cross-attentions for explainability if available
    # Note: not all models support this; adjust depending on your model
    attentions = None
    if hasattr(outputs, "cross_attentions"):
        attentions = outputs.cross_attentions  # List of attention tensors per layer

    return summary, attentions


def main():
    parser = argparse.ArgumentParser(description="Inference for Healthnotes Summarizer")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to a text file containing clinical notes, one per line.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="File path to save generated summaries, one per line.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/healthnotes-summarizer",
        help="Directory containing the fine-tuned model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max length for generated summaries.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cuda' or 'cpu').",
    )
    args = parser.parse_args()

    # Load tokenizer and model from the fine-tuned checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(args.device)

    # Read input clinical notes
    with open(args.input_file, "r", encoding="utf-8") as f:
        clinical_notes = [line.strip() for line in f if line.strip()]

    summaries = []
    for note in tqdm(clinical_notes, desc="Generating summaries"):
        summary, attentions = generate_summary(note, model, tokenizer, args.device, args.max_length)
        summaries.append(summary)
        # You can extend here to save or visualize attentions if needed

    # Save summaries to output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(summary + "\n")

    print(f"Inference complete. Summaries saved to {args.output_file}")


if __name__ == "__main__":
    main()
