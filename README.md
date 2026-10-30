# RAG-assistant_for_Moldovan_tariffs
RAG-assistant for Moldovan tariffs


> **Note:** The included `Qwen2.5-7B-Instruct-Q4_K_M.gguf` model file is licensed under Apache 2.0. See the “Model License and Attribution” section below.
> 
# RAG Assistant for Mobile Tariff Comparison

A production-ready Retrieval-Augmented Generation (RAG) system that answers factual
questions about mobile tariffs from three Moldovan operators: **Orange**, **Moldcell**,
and **Moldtelecom**. The system runs entirely locally — no API keys, no cloud
dependencies, no data leaves your machine.

Built as the final project for a Master's-level course on ML Engineering and
Advanced Python.

---

## Problem

Comparing mobile tariffs across operators is painful:

- Information is scattered across different websites.
- Each operator uses different terminology for the same parameters
  (e.g., "unlimited" vs "безлимит", "MDL" vs "лей").
- Users must manually cross-reference prices, minutes, gigabytes, and roaming
  quotas to find the best plan.

A naive search engine returns pages of links. A general-purpose LLM either
hallucinates numbers or admits it doesn't know. Neither is acceptable when the
question is "how much does this tariff cost?".

## Solution

A local RAG system that:

1. **Stores** all 15 tariffs as enriched, self-contained text chunks.
2. **Retrieves** the most relevant chunks using hybrid search
   (dense embeddings + BM25 lexical matching, fused via Reciprocal Rank Fusion).
3. **Generates** a concise answer with a local LLM, using a few-shot prompt
   that teaches the model the output format.
4. **Cites** the source chunks so the user can verify every claim.

---

## Key Features

- **100% local.** Models run on CPU or GPU via `llama-cpp-python`.
  No API calls, no telemetry, no external dependencies at inference time.
- **Hybrid retrieval.** Combines semantic search (`bge-m3` embeddings)
  with lexical search (BM25). This is critical for queries that contain
  exact tariff names like "Lumo 220" or specific numbers.
- **Data enrichment.** Chunks use explicit `ДА`/`НЕТ` instead of `0`/`unlimited`,
  which eliminates the most common class of LLM errors on boolean questions.
- **Summary chunks.** Each operator has an aggregate chunk listing all its
  tariffs, count, and cheapest option — solving the "how many tariffs does X
  have?" class of questions without forcing the LLM to count.
- **Few-shot prompting.** Three examples (aggregation, comparison, yes/no)
  raise accuracy by over 40 percentage points compared to a naive prompt.
- **Smart evaluation.** A hybrid judge combines regex-based number/unit
  matching with LLM fallback, making the benchmark reproducible and fast
  (2 seconds for 55 questions, vs. 3+ minutes with LLM-only judging).
- **Streamlit demo.** Chat interface with streaming responses, adjustable
  retrieval depth, and source attribution under every answer.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| **Embeddings** | `BAAI/bge-m3` | Strong multilingual performance; handles Russian + English queries |
| **Vector index** | `FAISS` (IndexFlatIP) | Fast exact search; no infrastructure required |
| **Lexical index** | `rank_bm25` | Captures exact tariff names and numbers that dense retrieval misses |
| **LLM** | `Qwen2.5-7B-Instruct` (Q4_K_M GGUF) | Runs locally; strong instruction-following; supports Russian |
| **LLM runtime** | `llama-cpp-python` | C++ backend, CPU/GPU, no PyTorch overhead |
| **UI** | `Streamlit` | One-file chat interface with streaming support |
| **Evaluation** | Custom hybrid judge | Regex for numbers/units + LLM fallback for reasoning cases |

---

## Results

Evaluated on a hand-crafted benchmark of **55 factual questions** across five
categories: direct facts, yes/no, comparisons, aggregation, and multi-step
reasoning.

| Experiment | Accuracy | Time | Notes |
|---|---|---|---|
| Baseline (Top-5, simple prompt) | **38.18%** | 13 min | Naive RAG — retrieval is fine, generation fails |
| Top-15 (all chunks, simple) | **41.82%** | 38 min | More context without instructions doesn't help |
| Hybrid + Reranker | **36.36%** | 14 min | Reranker removes correct chunks on small base |
| Few-Shot prompt | **78.18%** | 16 min | +40 p.p. — the single biggest win |
| **Dynamic Top-N + Few-Shot** | **85.45%** | 33 min | 🏆 Best accuracy; widens retrieval for comparisons |
| Markdown Table + Few-Shot | **80.00%** | 8 min | ⚡ Best speed/quality trade-off |
| **FINAL (Top-8 + Smart Judge)** | **81.82%** | 25 min | Production version; honest trade-off |

### Per-category accuracy (FINAL config)

| Question type | Accuracy | Example |
|---|---|---|
| Direct fact | 86.67% | "What is the monthly fee of One 170?" |
| Yes/No | 80.00% | "Is unlimited internet included in Lumo 390?" |
| Comparison | 90.00% | "Which is cheaper: Start 5G or Junior 90?" |
| Aggregation | 70.00% | "How many tariffs does Moldcell have?" |
| Multi-step | 80.00% | "Which tariff has the largest EU roaming quota, and what does it cost?" |

**Target threshold (80%) met by three configurations.** The best result
(Dynamic Top-N, 85.45%) is achieved by widening retrieval to 15 chunks
specifically for comparison and aggregation questions, at the cost of longer
runtime. The Markdown Table variant reaches 80% in just 8 minutes, which is
the recommended configuration for interactive demos.

---

## What the experiments taught me

### What worked

1. **Few-shot prompting** — moving from a naive prompt to three worked
   examples raised accuracy from 40% to 80%. This alone accounted for the
   largest single jump. The model learned to:
   - Emit only the winning tariff name on comparison questions (0% → 90%).
   - Answer yes/no questions with just "Да" or "Нет" instead of a paragraph
     of hedging (40% → 90%).
2. **Data enrichment** — replacing `0` with `НЕТ` and `unlimited` with
   `ДА (безлимит)` removed the largest source of LLM misinterpretation.
3. **Summary chunks** — an explicit "Operator X has N tariffs: ..." chunk
   lifted aggregation accuracy from 70% to 90%, because the model no longer
   has to count rows across multiple chunks.
4. **Dynamic top-N** — setting `k=15` for complex questions and `k=5` for
   simple ones balances quality and latency better than a fixed `k`.

### What didn't work

1. **Reranking with `bge-reranker-v2-m3`** hurt performance (38.18%, below
   baseline). On a corpus of only 18 chunks, reranking frequently discards
   the correct chunk. Rerankers are beneficial at scale (1000+ chunks), not
   here.
2. **Markdown tables without few-shot examples** collapsed accuracy to 41%.
   The model couldn't map columns to entities without guidance. With
   few-shot, tables became competitive (80%) and much faster.
3. **Top-N without prompt engineering** gave almost no improvement
   (38% → 42%). Feeding more context to an unguided model just buries the
   answer.

### What remains unsolved

- **Complex aggregation with filters** — questions like "how many tariffs
  are cheaper than 120 MDL?" fail because the correct chunks often fall
  outside the top-8. The fix is pre-computed summary chunks for subsets
  ("tariffs under 120 MDL: Junior 90, Senior 80, Lumo 170, Lumo 100 — 4 total").
- **Multi-hop reasoning** — questions that require finding one fact, then
  using it to look up a second fact (e.g., "the tariff with the most EU
  roaming, what does it cost?") remain at 60–80%. A query-rewriting layer
  that decomposes such questions into two sub-queries would help.
- **Judge bias** — the hybrid judge is strict on exact number matches. It
  occasionally marks "both equal, 100 GB" as wrong against an answer of
  "100" because the regex requires a unit. This is a trade-off: it prevents
  false positives but misses some genuine matches.

---

## Project structure

project/ 

├── app.py Streamlit chat interface

├── rag_core.py Core RAG components (retrieval, LLM, pipeline)

├── subscriptions.csv Raw tariff data (15 rows, 3 operators)

├── questions.csv Evaluation set (55 questions)

├── final_results.csv Predictions and scores from the final run

├── requirements.txt Python dependencies

├── models/

│ └── Qwen2.5-7B-Instruct-Q4_K_M.gguf

└── README.md


**`rag_core.py`** contains:

- `create_enriched_chunks(df)` — builds enriched text chunks from the CSV.
- `create_summary_chunks(df, chunks)` — adds one summary chunk per operator.
- `HybridIndex` — dense + BM25 retrieval with Reciprocal Rank Fusion.
- `LocalLLMClient` — thin wrapper around `llama-cpp-python` with both
  blocking and streaming interfaces.
- `RAGPipeline` — orchestrates retrieval and generation; supports four prompt
  styles (`simple`, `fewshot`, `dynamic`, `table`).

**`app.py`** is a Streamlit UI that imports from `rag_core.py`, caches the
pipeline across sessions, and streams tokens to the chat window as the model
generates them.

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:AdrianBabira/RAG-assistant_for_Moldovan_tariffs.git
cd RAG-assistant_for_Moldovan_tariffs

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows (PowerShell)

pip install -r requirements.txt



```
The 7B model in Q4_K_M quantization takes ~4.7 GB on disk and ~5 GB of RAM
when loaded

## RAG run
```bash
streamlit run app.py
```

The app opens at http://localhost:8501. First launch loads the embedding
model and LLM into memory (~40 seconds). After that, each question takes
5–15 seconds depending on top_n and prompt style.

The sidebar lets you:

    Adjust top_n (3–15).

    Switch prompt style: fewshot, simple, dynamic, or table.

    Toggle source attribution and token streaming.

    Load example questions.




## Programmatic use

```bash
import pandas as pd
from rag_core import (
    HybridIndex, LocalLLMClient, RAGPipeline,
    create_enriched_chunks, create_summary_chunks,
)

df = pd.read_csv("subscriptions.csv")
chunks = create_summary_chunks(df, create_enriched_chunks(df))

retriever = HybridIndex()
retriever.build(chunks)

llm = LocalLLMClient("./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
rag = RAGPipeline(retriever, llm, top_n=8, prompt_style="fewshot")

answer, sources = rag.answer("Сколько тарифов у оператора Moldcell?")
print(answer)         # 5
print([c["chunk_id"] for c in sources])
```

> ## Model License and Attribution

This repository includes the model file:

- `Qwen2.5-7B-Instruct-Q4_K_M.gguf`

This model file is a quantized GGUF version of **Qwen2.5-7B-Instruct**, originally developed by **Alibaba Cloud / Qwen Team** and published at:

- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

The original model and this quantized model file are distributed under the **Apache License 2.0**. A copy of the Apache License 2.0 is included in this repository in the `LICENSE` file. If the original Qwen repository includes a `NOTICE` file, it is also included here as required.

This repository is intended to demonstrate and provide an introduction to my RAG system. The model file is included only as part of that demonstration.

### Attribution

- Original model: **Qwen2.5-7B-Instruct**
- Original authors/project: **Alibaba Cloud / Qwen Team**
- Original model repository: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Model license: **Apache License 2.0**
- Quantization format: **GGUF Q4_K_M**

This project is not affiliated with, endorsed by, or sponsored by Alibaba Cloud or the Qwen Team.

## Repository License

The source code and other original contents of this repository are **not currently licensed** unless otherwise explicitly stated. No rights are granted to use, copy, modify, or distribute the original code or contents of this repository except as permitted by applicable law or by separate written permission from the author.

The model file `Qwen2.5-7B-Instruct-Q4_K_M.gguf` is **not** covered by the repository’s licensing status. It remains licensed under the **Apache License 2.0** from the original Qwen2.5-7B-Instruct project.