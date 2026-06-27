# MedNotes RAG — Clinical Document QA Assistant

A Retrieval-Augmented Generation system for answering questions over de-identified clinical notes.

## Features

- Hybrid Retrieval:
  - HuggingFace sentence-transformer embeddings
  - ChromaDB vector search
  - BM25 keyword retrieval

- Reranking:
  - CrossEncoder reranker improves retrieval relevance

- Multi-agent architecture:
  - Query Agent
  - Retriever Agent
  - Clinical Summarizer Agent

Built using CrewAI role-based orchestration.

## Architecture

User Question

↓

Query Agent

↓

Hybrid Retriever
(ChromaDB + BM25)

↓

CrossEncoder Reranking

↓

Summarizer Agent

↓

Grounded Clinical Answer + Sources


## Tech Stack

- Python
- CrewAI
- ChromaDB
- HuggingFace Transformers
- Sentence Transformers
- Streamlit


## Evaluation

Evaluation pipeline checks:

- Retrieval quality
- Answer grounding
- Keyword evidence matching

Run:

python evaluations/evaluate_rag.py


## Run Application

Install:

pip install -r requirements.txt


Start:

streamlit run app.py


## Project Structure

src/
 ├── agents/
 ├── tools/
 ├── memory/
 ├── crew.py

evaluations/
 ├── test_dataset.json
 └── evaluate_rag.py


## Goals

Reduce hallucination by forcing answers to use retrieved clinical evidence.