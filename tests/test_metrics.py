import pytest
from src.metrics import compute_rouge, compute_bertscore

def test_compute_rouge():
    preds = ["The patient was discharged in good condition."]
    refs = ["Patient discharged healthy after treatment."]
    scores = compute_rouge(preds, refs)
    assert "rouge1_f1" in scores
    assert 0.0 <= scores["rouge1_f1"] <= 1.0

def test_compute_bertscore():
    preds = ["The patient was discharged in good condition."]
    refs = ["Patient discharged healthy after treatment."]
    f1 = compute_bertscore(preds, refs)
    assert 0.0 <= f1 <= 1.0
