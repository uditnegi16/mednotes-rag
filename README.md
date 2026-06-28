<p align="center">
  <img width="100%" alt="MedNotes RAG Banner" src="docs/banner.png"/>
</p>

<p align="center">
  <a href="YOUR_STREAMLIT_DEMO_URL"><img src="https://img.shields.io/badge/Live_Demo-Streamlit-success?style=flat-square" alt="Live Demo"/></a>
  <a href="https://github.com/uditnegi16/MedNotes-RAG"><img src="https://img.shields.io/badge/GitHub-MedNotes_RAG-181717?style=flat-square&logo=github"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/CrewAI-Multi--Agent-orange?style=flat-square"/></a>
  <a href="#"><img src="https://img.shields.io/badge/ChromaDB-Vector_DB-5E17EB?style=flat-square"/></a>
  <a href="#"><img src="https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=flat-square"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/RAG-Clinical_QA-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Evaluation-Keyword_Coverage-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Hallucination-Grounded_Answers-important?style=flat-square"/>
  <img src="https://img.shields.io/badge/CI-GitHub_Actions-black?style=flat-square"/>
</p>

---

# MedNotes RAG — Clinical Document QA Assistant

Production-style Retrieval-Augmented Generation (RAG) system for answering questions over de-identified clinical documents using a **CrewAI multi-agent pipeline**, **HuggingFace embeddings**, **CrossEncoder reranking**, **ChromaDB vector search**, and an **evaluation framework** measuring retrieval quality and answer grounding.

Unlike a simple chatbot, MedNotes separates retrieval and answer generation into dedicated agents while validating system quality through automated evaluation before deployment.

---

# Overview

MedNotes RAG allows users to ask natural language medical questions over a corpus of de-identified clinical notes.

The system first retrieves the most relevant documents using semantic embeddings, reranks them with a CrossEncoder, and finally generates a grounded response using only retrieved evidence.

A CrewAI workflow orchestrates independent agents responsible for:

- Query optimization
- Clinical document retrieval
- Evidence-based summarization

An integrated evaluation pipeline measures retrieval quality, keyword coverage, answer grounding and overall system performance to support an SDLC-style workflow instead of relying only on manual testing.

---

# Demo

YOUR_GITHUB_VIDEO_URL_HERE

---

# Screenshots

<p align="center">
  <img src="screenshots/home.png" width="48%">
  <img src="screenshots/chat.png" width="48%">
</p>

<p align="center">
  <img src="screenshots/history.png" width="48%">
  <img src="screenshots/evaluation.png" width="48%">
</p>

---

# System Architecture

<p align="center">
  <img width="90%" src="docs/architecture.png">
</p>

---

# Multi-Agent Workflow

```mermaid
flowchart LR

A[User Question]

A --> B[Query Agent]

B --> C[Retriever Agent]

C --> D[Sentence Transformer Embedding]

D --> E[ChromaDB Vector Search]

E --> F[CrossEncoder Re-ranking]

F --> G[Summarizer Agent]

G --> H[Grounded Clinical Answer]

H --> I[Evaluation Pipeline]
```

---

# Tech Stack

| Layer | Technology |
|--------|------------|
| Frontend | Streamlit |
| Agent Framework | CrewAI |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | ChromaDB |
| Retrieval | Semantic Search |
| Re-ranking | CrossEncoder |
| LLM | HuggingFace Inference API |
| Evaluation | Custom Python Evaluation Suite |
| Dataset | MTSamples Clinical Notes |
| Logging | Python Logging |
| Language | Python 3.12 |

---

# Features

- CrewAI multi-agent orchestration
- Dedicated Query Agent
- Dedicated Retriever Agent
- Dedicated Summarizer Agent
- Semantic document retrieval
- HuggingFace embeddings
- ChromaDB vector database
- CrossEncoder reranking
- Clinical document question answering
- Source-grounded responses
- Streamlit interface
- Persistent chat history
- Automatic conversation titles
- Logging support
- Evaluation framework
- GitHub Actions CI
- Modular project structure

---

# Evaluation Pipeline

Unlike basic portfolio RAG projects, MedNotes includes an automated evaluation suite.

The evaluation measures:

- Retrieval Coverage
- Keyword Coverage
- Source Grounding
- Successful Retrieval Rate
- Overall Evaluation Score

The evaluation dataset consists of multiple manually designed clinical QA pairs covering different medical specialties.

---

# Sample Evaluation Results

| Metric | Result |
|---------|---------|
| Tests Executed | 35 |
| Retrieval Coverage | 83.33% |
| Source Grounding | 100% |
| Overall Evaluation | 63.97% |
| Failed Queries | Mostly due to HuggingFace monthly API credit exhaustion rather than retrieval failures |

> The grounding metric verifies that generated answers stay within retrieved clinical evidence instead of introducing unsupported medical facts.

---

# Project Structure

```text
MedNotes-RAG/

├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── agents/
│   ├── memory/
│   ├── tools/
│   ├── crew.py
│   ├── logger.py
│   └── config.py
│
├── data/
│   ├── documents/
│   └── vectorstore/
│
├── evaluations/
│   ├── evaluate_rag.py
│   ├── test_dataset.json
│   └── embedding_queries.json
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── screenshots/
```

---

# Local Setup

## Clone Repository

```bash
git clone https://github.com/uditnegi16/MedNotes-RAG.git

cd MedNotes-RAG
```

## Create Virtual Environment

```powershell
python -m venv .venv

.venv\Scripts\activate
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Create `.env`

```env
HF_TOKEN=your_huggingface_token

LLM_MODEL=your_model_name
```

## Build Vector Database

```powershell
python ingest.py
```

## Run Streamlit

```powershell
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

# Running Evaluation

```powershell
python evaluations/evaluate_rag.py
```

The evaluation automatically:

- Runs every question
- Retrieves evidence
- Generates answers
- Calculates keyword coverage
- Performs source-grounding checks
- Produces a final evaluation report

---

# Continuous Integration

GitHub Actions automatically executes on every push:

- Dependency installation
- Import validation
- Evaluation script execution
- Build verification
- Basic regression testing

---
