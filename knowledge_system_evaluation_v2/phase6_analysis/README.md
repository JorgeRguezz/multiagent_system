# Phase 6 Analysis

This folder contains local post-processing analyses for the IEEE Access evaluation.

The first script computes paired bootstrap confidence intervals for the main comparison:

`Graph-RAG - Vector-only RAG`

It does not rerun inference, RAGAS, BERTScore, or any LLM judge. It only reads the completed per-question evaluation file:

`knowledge_system_evaluation_v2/evaluated_datasets/community_qa_dataset_evaluated.json`

## Run

From the project root:

```bash
python3 knowledge_system_evaluation_v2/phase6_analysis/bootstrap_graph_vs_vector.py
```

Optional arguments:

```bash
python3 knowledge_system_evaluation_v2/phase6_analysis/bootstrap_graph_vs_vector.py \
  --iterations 10000 \
  --seed 20260709
```

If the evaluated dataset is elsewhere, pass it explicitly:

```bash
python3 knowledge_system_evaluation_v2/phase6_analysis/bootstrap_graph_vs_vector.py \
  --input /path/to/community_qa_dataset_evaluated.json
```

## Outputs

The script writes:

- `knowledge_system_evaluation_v2/phase6_analysis/bootstrap_graph_vs_vector.json`
- `knowledge_system_evaluation_v2/phase6_analysis/bootstrap_graph_vs_vector.csv`

Each row contains the observed means, observed paired difference, bootstrap 95% confidence interval, and the fraction of bootstrap samples where the difference is above zero.

