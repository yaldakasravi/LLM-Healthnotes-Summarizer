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
Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Download MIMIC-III clinical notes and summaries or use provided sample data.
Training

Fine-tune the summarization model with:

python src/train.py --config configs/training_config.yaml
Inference & Explainability

Generate summaries with attention weights using:

python src/infer.py --input_file data/sample_notes.txt --output_file outputs/summaries.txt
Gradio App

Launch the interactive web interface:

python app.py
Evaluation

Evaluate model summaries with ROUGE and BERTScore:

python src/metrics.py --pred summaries.txt --ref references.txt
Comparison with GPT-4

Benchmark zero-shot GPT-4 summaries (requires OpenAI API key):

python src/gpt4_eval.py --input_file data/sample_notes.txt --output_file outputs/gpt4_summaries.txt
License

MIT License

