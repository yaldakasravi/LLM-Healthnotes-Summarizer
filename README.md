# LLM Healthnotes Summarizer

This project fine-tunes a transformer-based summarization model to convert patient clinical notes from the MIMIC-III dataset into concise, human-readable discharge summaries.

## Features

- Fine-tuning of DistilBART or Flan-T5 models on clinical notes.
- Interactive Gradio interface to upload clinical notes and get summaries.
- Token-level attention visualization for explainability.
- Benchmarking with ROUGE and BERTScore metrics.
- Comparison with zero-shot GPT-4 summarization.

## Dataset

- Uses the publicly available MIMIC-III dataset (requires data access approval).
- Sample subset of notes and summaries included for demo/testing purposes.

## Setup

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd LLM_Healthnotes_Summarizer
