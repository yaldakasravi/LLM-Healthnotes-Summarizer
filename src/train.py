import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from src.utils import preprocess_data  # We'll implement this helper function separately

def main():
    # Load config from YAML file
    with open("configs/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    set_seed(config.get("seed", 42))

    # Load tokenizer and model from pretrained checkpoint
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    # Load dataset from CSV files, expects 'clinical_note' and 'discharge_summary' columns
    dataset = load_dataset(
        "csv",
        data_files={
            "train": config["train_data_path"],
            "validation": config["val_data_path"],
        }
    )

    # Preprocess dataset (tokenization + formatting)
    tokenized_datasets = dataset.map(
        lambda batch: preprocess_data(batch, tokenizer, config),
        batched=True,
        remove_columns=["clinical_note", "discharge_summary"]
    )

    # Prepare data collator for dynamic padding during batching
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy=config.get("evaluation_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_train_epochs"],
        logging_dir=config.get("logging_dir", "./logs"),
        logging_steps=config.get("logging_steps", 100),
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the final model checkpoint
    trainer.save_model(config["output_dir"])

    print(f"Training complete. Model saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
