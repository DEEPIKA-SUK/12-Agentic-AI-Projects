# Project 2: RAG Agent with Citation Grounding

A production-grade Retrieval-Augmented Generation (RAG) agent featuring **in-memory vector search**, **strict inline citation grounding**, **hallucination verification guardrails**, and **low-confidence fallback routing**.

---

## 📌 Features

- **Zero-Dependency Vector Retrieval:** Built-in `LocalVectorIndex` utilizing **TF-IDF** and **Cosine Similarity** to index and retrieve relevant document chunks without needing external vector database servers.
- **Strict Citation Grounding:** Answers are generated with explicit source tracking tags (e.g., `[DOC-101]`) mapping directly back to underlying document snippets.
- **Citation Validator Guardrail:** Uses regex-based pattern matching to cross-check generated citation tags against retrieved context to prevent hallucinated references.
- **Low-Confidence Fallback Trigger:** Automatically flags low-relevance queries or uncited claims to safely halt ungrounded responses.
- **Pydantic v2 Schema Enforcement:** Type-safe definitions for document chunks, citations, and agent responses.

---

## 🏗 Architecture & Workflow

```text
 ┌────────────────┐     1. Search Query      ┌─────────────────────────┐
 │ User Question  │ ───────────────────────► │ Local Vector Index      │
 └────────────────┘                          │ (TF-IDF + Cosine Sim)   │
                                             └────────────┬────────────┘
                                                          │
                                             2. Top-K Chunks & Scores
                                                          │
                                                          ▼
 ┌────────────────┐     3. Verification      ┌─────────────────────────┐
 │ Fallback Engine│ ◄─────────────────────── │ Grounded Generator      │
 └────────────────┘    Failed / Low Score    │ (Synthesizes & Cites)   │
         ▲                                   └────────────┬────────────┘
         │                                                │
         │             4. Invalid Tag Detected            │
         └────────────────────────────────────────────────┘
```
---

## Project Structure
```
project-2/
├── requirements.txt   # Core Python dependencies
├── main.py            # Complete agent implementation & test suite
└── README.md          # Project documentation
```

## Commands
```
pip install -r requirements.txt
python main.py
```

## Output
```
User Query: 'What is the return policy duration?'
2026-09-27 18:31:00 [INFO] RAGAgent - Processing Query: 'What is the return policy duration?'
2026-09-27 18:31:00 [INFO] RAGAgent - Retrieved 2 relevant document chunks.
Answer:     Customers can return any item within 30 days of purchase for a full refund [DOC-101].
Confidence: 0.42
Fallback:   False
Citations:
  - [DOC-101] Snippet: 'return any item within 30 days of purchase for a full refund'
-----------------------------------------------------------------

User Query: 'What is the policy for quantum satellite communication hardware?'
2026-09-27 18:31:00 [INFO] RAGAgent - Processing Query: 'What is the policy for quantum satellite communication hardware?'
2026-09-27 18:31:00 [WARNING] RAGAgent - No relevant context found above similarity threshold. Triggering fallback.
Answer:     I am sorry, but I do not have sufficient internal information to answer this question accurately.
Confidence: 0.0
Fallback:   True
-----------------------------------------------------------------
```
