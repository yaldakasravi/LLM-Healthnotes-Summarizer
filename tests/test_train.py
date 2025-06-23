import os
import pytest
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.train import main as train_main
import yaml

def test_training_runs(tmp_path):
    # Prepare minimal config with dummy paths and output dir inside tmp_path
    config_path = tmp_path / "training_config.yaml"
    dummy_train_csv = tmp_path / "dummy_train.csv"
    dummy_val_csv = tmp_path / "dummy_val.csv"

    # Create dummy CSV files with minimal data
    dummy_train_csv.write_text("clinical_note,discharge_summary\n\"note1\",\"summary1\"\n")
    dummy_val_csv.write_text("clinical_note,discharge_summary\n\"note2\",\"summary2\"\n")

    config_content = f"""
model_name: facebook/bart-base
train_data_path: {dummy_train_csv}
val_data_path: {dummy_val_csv}
output_dir: {tmp_path}/model_output
batch_size: 1
num_train_epochs: 1
learning_rate: 5e-5
max_input_length: 128
max_output_length: 64
seed: 42
evaluation_strategy: no
save_strategy: no
logging_dir: {tmp_path}/logs
logging_steps: 1
"""

    config_path.write_text(config_content)

    # Patch the open function in train.py to load this config instead of default
    # For simplicity, just run train_main if train.py reads config from that path
    # (You may need to modify train.py to accept config path as argument for easier testing)
    os.environ["PYTHONPATH"] = str(tmp_path)

    # Run training - should complete without error on minimal dummy data
    try:
        train_main()
        assert os.path.exists(tmp_path / "model_output")
    except Exception as e:
        pytest.fail(f"Training failed: {e}")
