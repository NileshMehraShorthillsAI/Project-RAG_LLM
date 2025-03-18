import json
from tqdm import tqdm
from evaluate import load
from transformers import pipeline
from statistics import mean
import bert_score

# Load evaluation metrics
bleu = load("bleu")
meteor = load("meteor")
rouge = load("rouge")
bertscore = load("bertscore")

# Initializing NLI model for context relevance and faithfulness evaluation
nli_model = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

def compute_nli_score(reference, hypothesis):
    """
    Compute NLI-based score (used for both context relevance and faithfulness).
    Returns a score between 0 and 1 indicating relevance.
    """
    result = nli_model(f"{reference} [SEP] {hypothesis}")
    score = result[0]['score'] if result[0]['label'] == 'ENTAILMENT' else 1 - result[0]['score']
    return score

# Load ground truth and RAG-generated QA pairs
with open("generated_ques_ans.json", "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

with open("RAG_generated_qa_pairs.json", "r", encoding="utf-8") as f:
    rag_data = json.load(f)

assert len(ground_truth_data) == len(rag_data), "Mismatch in number of QA pairs between the two files."

evaluation_results = []

# Lists to store individual metric scores
bleu_scores = []
meteor_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
bertscore_f1_scores = []
context_relevance_scores = []
faithfulness_scores = []

for gt_pair, rag_pair in tqdm(zip(ground_truth_data, rag_data), total=len(ground_truth_data), desc="Evaluating QA Pairs"):
    question = gt_pair["question"]
    ground_truth_answer = gt_pair["answer"]
    generated_answer = rag_pair["answer"]

    # Compute BLEU score
    bleu_score = bleu.compute(predictions=[generated_answer], references=[ground_truth_answer])['bleu']
    bleu_scores.append(bleu_score)

    # Compute METEOR score
    meteor_score = meteor.compute(predictions=[generated_answer], references=[ground_truth_answer])['meteor']
    meteor_scores.append(meteor_score)

    # Compute ROUGE scores
    rouge_scores = rouge.compute(predictions=[generated_answer], references=[ground_truth_answer])
    rouge1_scores.append(rouge_scores['rouge1'])
    rouge2_scores.append(rouge_scores['rouge2'])
    rougeL_scores.append(rouge_scores['rougeL'])

    # Compute BERTScore F1
    bertscore_result = bertscore.compute(predictions=[generated_answer], references=[ground_truth_answer], lang="en")
    bertscore_f1 = bertscore_result['f1'][0]  # Extract F1 score
    bertscore_f1_scores.append(bertscore_f1)

    # Compute Context Relevance
    context_relevance_score = compute_nli_score(ground_truth_answer, generated_answer)
    context_relevance_scores.append(context_relevance_score)

    # Compute Faithfulness
    faithfulness_score = compute_nli_score(generated_answer, ground_truth_answer)  # Reverse order for faithfulness
    faithfulness_scores.append(faithfulness_score)

    # Append results
    evaluation_results.append({
        "Question": question,
        "Ground Truth": ground_truth_answer,
        "Generated Answer": generated_answer,
        "Metrics": {
            "BLEU Score": bleu_score,
            "METEOR Score": meteor_score,
            "ROUGE-1": rouge_scores['rouge1'],
            "ROUGE-2": rouge_scores['rouge2'],
            "ROUGE-L": rouge_scores['rougeL'],
            "BERTScore F1": bertscore_f1,
            "Context Relevance": context_relevance_score,
            "Faithfulness": faithfulness_score
        }
    })

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

# Calculate and print average scores
average_bleu = mean(bleu_scores)
average_meteor = mean(meteor_scores)
average_rouge1 = mean(rouge1_scores)
average_rouge2 = mean(rouge2_scores)
average_rougeL = mean(rougeL_scores)
average_bertscore_f1 = mean(bertscore_f1_scores)
average_context_relevance = mean(context_relevance_scores)
average_faithfulness = mean(faithfulness_scores)

print(f"✅ Evaluation completed! Results saved to 'evaluation_results.json'.")
print(f"\nAverage Scores:")
print(f"BLEU Score: {average_bleu:.4f}")
print(f"METEOR Score: {average_meteor:.4f}")
print(f"ROUGE-1: {average_rouge1:.4f}")
print(f"ROUGE-2: {average_rouge2:.4f}")
print(f"ROUGE-L: {average_rougeL:.4f}")
print(f"BERTScore F1: {average_bertscore_f1:.4f}")
print(f"Context Relevance: {average_context_relevance:.4f}")
print(f"Faithfulness: {average_faithfulness:.4f}")