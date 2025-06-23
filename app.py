import gradio as gr
import torch
from src.model_utils import load_model_and_tokenizer, extract_token_level_attention
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model and tokenizer once at startup
model_dir = "models/healthnotes-summarizer"  # Update if needed
model, tokenizer, device = load_model_and_tokenizer(model_dir)

def summarize_and_explain(clinical_note):
    """
    Generates summary and attention visualizations for a given clinical note.

    Args:
        clinical_note (str): Input clinical note text.

    Returns:
        summary (str): Generated summary text.
        attention_fig (matplotlib.figure.Figure): Attention heatmap figure.
    """
    if not clinical_note.strip():
        return "Please enter clinical notes to summarize.", None

    summary, attentions, input_tokens, summary_tokens = extract_token_level_attention(
        clinical_note, model, tokenizer, device
    )

    if attentions is None:
        # If model does not output attentions
        return summary, None

    # We'll visualize the cross-attention weights from the last decoder layer and first head
    # attentions shape: tuple of layers, each is tensor (batch, heads, tgt_len, src_len)
    last_layer_attention = attentions[-1][0]  # shape: (heads, tgt_len, src_len)
    head_attention = last_layer_attention[0].cpu().numpy()  # pick first head

    # Crop to actual sequence lengths (remove padding tokens)
    tgt_len, src_len = head_attention.shape
    input_tokens_trimmed = input_tokens[:src_len]
    summary_tokens_trimmed = summary_tokens[:tgt_len]

    # Plot heatmap
    plt.figure(figsize=(min(15, src_len/2), min(8, tgt_len/2)))
    sns.heatmap(head_attention, xticklabels=input_tokens_trimmed, yticklabels=summary_tokens_trimmed,
                cmap="viridis", cbar=True)
    plt.xlabel("Input Tokens")
    plt.ylabel("Summary Tokens")
    plt.title("Cross-Attention (Decoder->Encoder) Heatmap (Last Layer, Head 1)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig = plt.gcf()
    plt.close()

    return summary, fig

# Gradio interface
iface = gr.Interface(
    fn=summarize_and_explain,
    inputs=gr.Textbox(lines=15, label="Enter Clinical Notes"),
    outputs=[
        gr.Textbox(lines=5, label="Generated Summary"),
        gr.Image(type="pil", label="Attention Visualization")
    ],
    title="Healthnotes Summarizer with Explainability",
    description="Paste clinical notes to get a discharge summary with token-level attention visualization."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
