import argparse
from rouge_score import rouge_scorer, scoring
from bert_score import score as bert_score
import logging

def compute_rouge(predictions, references):
    """
    Compute average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    
    Args:
        predictions (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.
    
    Returns:
        dict: Average ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    avg_scores = {
        "rouge1_f1": result["rouge1"].mid.fmeasure,
        "rouge2_f1": result["rouge2"].mid.fmeasure,
        "rougeL_f1": result["rougeL"].mid.fmeasure,
    }
    return avg_scores

def compute_bertscore(predictions, references, lang="en"):
    """
    Compute average BERTScore F1.
    
    Args:
        predictions (List[str]): List of generated summaries.
        references (List[str]): List of reference summaries.
        lang (str): Language code, default "en".
    
    Returns:
        float: Average BERTScore F1.
    """
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=True)
    return F1.mean().item()

def load_text_file(filepath):
    """
    Load lines from a text file.
    Strips newlines and ignores empty lines.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries with ROUGE and BERTScore.")
    parser.add_argument(
        "--pred", type=str, required=True,
        help="File containing predicted/generated summaries, one per line."
    )
    parser.add_argument(
        "--ref", type=str, required=True,
        help="File containing reference summaries, one per line."
    )
    parser.add_argument(
        "--lang", type=str, default="en",
        help="Language code for BERTScore (default: en)."
    )

    args = parser.parse_args()

    predictions = load_text_file(args.pred)
    references = load_text_file(args.ref)

    if len(predictions) != len(references):
        logging.error(f"Number of predictions ({len(predictions)}) does not match references ({len(references)})!")
        return

    rouge_scores = compute_rouge(predictions, references)
    bert_f1 = compute_bertscore(predictions, references, lang=args.lang)

    print("\nEvaluation Results:")
    print(f"ROUGE-1 F1 Score: {rouge_scores['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1 Score: {rouge_scores['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1 Score: {rouge_scores['rougeL_f1']:.4f}")
    print(f"BERTScore F1: {bert_f1:.4f}")

if __name__ == "__main__":
    main()
