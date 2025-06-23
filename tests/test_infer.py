import pytest
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.infer import generate_summary

@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def test_generate_summary(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    input_text = "Patient presented with fever and cough."
    device = "cpu"
    summary, attentions = generate_summary(input_text, model, tokenizer, device)
    assert isinstance(summary, str)
    # Since model is base and not fine-tuned, summary may be default but must be string
    assert len(summary) > 0
