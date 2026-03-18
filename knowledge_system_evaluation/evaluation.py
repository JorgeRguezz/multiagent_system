import os
import json
import pandas as pd
from datasets import Dataset
import evaluate as hf_evaluate

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "[API_KEY]"

llm_judge = llm_factory("gpt-4o-mini")

embeddings_model = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
)

print(hasattr(embeddings_model, "embed_query"))
print(hasattr(embeddings_model, "embed_documents"))
print(hasattr(embeddings_model, "embed_text"))

def run_ragas(df, answer_col):
    eval_df = df[["question", "context", "answer_gold", answer_col]].copy()
    eval_df.columns = ["question", "contexts", "ground_truth", "answer"]

    eval_df["contexts"] = eval_df["contexts"].apply(
        lambda x: [x] if isinstance(x, str) else (x if isinstance(x, list) else [])
    )

    def trim_text(x, max_chars):
        if not isinstance(x, str):
            return ""
        return x[:max_chars]

    eval_df["question"] = eval_df["question"].apply(lambda x: trim_text(x, 600))
    eval_df["answer"] = eval_df["answer"].apply(lambda x: trim_text(x, 1800))
    eval_df["ground_truth"] = eval_df["ground_truth"].apply(lambda x: trim_text(x, 1800))
    eval_df["contexts"] = eval_df["contexts"].apply(
        lambda lst: [trim_text(t, 2500) for t in lst[:1]]
    )

    dataset = Dataset.from_pandas(eval_df, preserve_index=False)

    metrics = [
        Faithfulness(llm=llm_judge),
        AnswerRelevancy(llm=llm_judge),
        AnswerCorrectness(llm=llm_judge),
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
        embeddings=embeddings_model,
        raise_exceptions=True,
    )

    return result.to_pandas().mean(numeric_only=True)

INPUT_FILE = "/home/gatv-projects/Desktop/project/knowledge_system_evaluation/ragas_eval_report.json"
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)

rouge = hf_evaluate.load("rouge")
bertscore = hf_evaluate.load("bertscore")

def compute_classic_metrics(predictions, references):
    rouge_results = rouge.compute(predictions=predictions, references=references)
    bs_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="distilbert-base-uncased",
    )
    return {
        "rougeL": rouge_results["rougeL"],
        "bert_f1": sum(bs_results["f1"]) / len(bs_results["f1"])
    }

systems = ["rag_answer", "strong_llm_answer", "weak_llm_answer"]
final_comparison = []

for sys in systems:
    print(f"\n>>> Evaluating system: {sys}...")

    classic = compute_classic_metrics(
        df[sys].fillna("").tolist(),
        df["answer_gold"].fillna("").tolist()
    )

    ragas_res = run_ragas(df.fillna(""), sys)

    combined = {
        "System": sys,
        "ROUGE-L": classic["rougeL"],
        "BERTScore-F1": classic["bert_f1"],
        "Faithfulness": ragas_res.get("faithfulness", ragas_res.get("Faithfulness")),
        "Relevancy": ragas_res.get("answer_relevancy", ragas_res.get("AnswerRelevancy")),
        "Correctness": ragas_res.get("answer_correctness", ragas_res.get("AnswerCorrectness")),
    }
    final_comparison.append(combined)

report_df = pd.DataFrame(final_comparison)
print("\n=== FINAL COMPARISON TABLE (LEADERBOARD) ===")
print(report_df.to_string(index=False))
report_df.to_csv("evaluation_leaderboard.csv", index=False)