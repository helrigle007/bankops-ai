# BankOps AI

**Production-pattern agentic AI for commercial banking customer service** — multi-agent pipeline with structured classification, deterministic escalation, RAG-grounded policy retrieval, and tool-calling agents, built on LangGraph.

---

## What This Demonstrates

- **Structured output classification** — Pydantic-validated intent detection with confidence scores and entity extraction, not free-text parsing
- **Deterministic HITL escalation** — Pure Python rule engine for confidence/risk-based routing to human review queues, fully auditable for compliance
- **RAG with policy grounding** — Markdown-aware chunking preserves banking policy structure; ChromaDB vector retrieval with local HuggingFace embeddings
- **Tool-calling agents** — ReAct agents bound to banking tools (account lookup, wire initiation, case management) via LangGraph supervisor orchestration
- **Golden dataset evaluation** — 10-scenario eval suite scoring intent accuracy, entity extraction, and escalation routing
- **LangSmith observability** — Full trace capture across classification, retrieval, tool calls, and escalation decisions

---

## Eval Results

10/10 pass rate across all metrics on the golden dataset:

| Metric | Score |
|--------|-------|
| Intent Classification | 10/10 (100%) |
| Escalation Routing | 10/10 (100%) |
| Entity Extraction | 10/10 (100%) |
| **Overall** | **10/10 (100%)** |

```bash
python main.py --eval    # Run it yourself
```

---

## Architecture

```
Customer Request
       │
       ▼
┌─────────────────┐
│   Classifier     │  Intent detection + entity extraction
│  (Structured)    │  → account_id, amount, urgency, confidence
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│   Escalation     │  Confidence + risk threshold evaluation
│    Engine        │  → auto_resolve | human_review | urgent_escalation
└───────┬─────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────────────┐
│ Agent │  │  Human Queue  │
│ Route │  │  (HITL)       │
└──┬───┘  └──────────────┘
   │
   ├── Wire Transfer Agent (tool calling)
   ├── Dispute Agent (tool calling)
   ├── Account Agent (tool calling)
   ├── Policy Agent (RAG grounded)
   │
   ▼
┌─────────────────┐
│  Structured      │  Response + audit metadata
│  Output          │  → tools called, confidence, escalation path
└─────────────────┘
```

## Key Components

**Intent Classification** — Structured LLM output that classifies requests into 6 intent categories with confidence scores, entity extraction (account IDs, amounts, case numbers), and urgency assessment.

**RAG Policy Agent** — Retrieval-augmented generation over banking policy documents (wire transfers, disputes, account maintenance). Uses ChromaDB for vector storage with markdown-aware chunking that preserves document structure.

**LangGraph Supervisor** — Planner/supervisor pattern that orchestrates the full pipeline. Classifies → evaluates escalation → routes to specialized sub-agents → assembles structured output with audit trail.

**Tool-Calling Agents** — Sub-agents bound to simulated banking tools (account lookup, transaction history, wire initiation, case management, compliance checks). In production these integrate with CRM, core banking, and case management systems.

**Human-in-the-Loop Escalation** — Layered rule engine that evaluates confidence scores, dollar amounts, account status, and risk indicators to determine whether to auto-resolve or route to human review queues with configurable SLAs.

**Evaluation Framework** — Golden dataset with 10 test scenarios covering intent accuracy, entity extraction, and escalation routing. Outputs scored results with pass/fail and aggregate metrics.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Deterministic escalation engine** | Pure Python rule engine, not LLM-driven. Thresholds are auditable, configurable without prompt changes, and compliance can review rules independently. |
| **Structured output classification** | `with_structured_output()` returns Pydantic models, giving type-safe confidence scores and entities that downstream logic can rely on. |
| **Markdown-aware RAG chunking** | Banking policies are hierarchical — the text splitter uses markdown separators to keep sections intact rather than splitting mid-thought. |
| **LangGraph over vanilla LangChain** | Conditional routing (escalation vs. auto-resolve) is cleaner as a graph than nested agent calls, with testable state at each node. |

See [docs/architecture.md](docs/architecture.md) for full design rationale and production considerations.

## Tech Stack

- **LangChain** / **LangGraph** — Agent orchestration and graph-based workflows
- **LangSmith** — Tracing, observability, and eval tracking
- **ChromaDB** — Vector storage for RAG retrieval
- **Anthropic Claude** — LLM (claude-sonnet-4-6)
- **HuggingFace** — Embeddings (all-MiniLM-L6-v2, runs locally)
- **Pydantic** — Structured outputs and data validation
- **Rich** — Terminal output formatting

## Setup

**Prerequisites:** Python 3.11+, an [Anthropic API key](https://console.anthropic.com/), and optionally a [LangSmith API key](https://smith.langchain.com/) for tracing.

```bash
# Clone and install
git clone https://github.com/helrigle007/bankops-ai.git
cd bankops-ai
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
# Optionally add LANGCHAIN_API_KEY for LangSmith tracing

# Build the knowledge base
python main.py --ingest

# Run the interactive demo
python main.py

# Run evaluation suite
python main.py --eval

# Process a single request
python main.py --request "What's the balance on account ACC-10042?"
```

## Project Structure

```
bankops-ai/
├── main.py                          # Entry point (demo, eval, ingest, single request)
├── agents/
│   ├── supervisor.py                # LangGraph supervisor workflow
│   ├── classification.py            # Intent classification + entity extraction
│   ├── policy_agent.py              # RAG-grounded policy Q&A
│   └── escalation.py                # Human-in-the-loop escalation engine
├── tools/
│   └── banking_tools.py             # Simulated banking tool integrations
├── knowledge/
│   ├── policies/                    # Sample banking policy documents
│   │   ├── wire_transfer_policy.md
│   │   ├── dispute_resolution.md
│   │   └── account_maintenance.md
│   └── ingest.py                    # Knowledge ingestion pipeline
├── evals/
│   ├── golden_dataset.json          # Test cases with expected outcomes
│   └── eval_runner.py               # Evaluation framework and scoring
└── docs/
    └── architecture.md              # Detailed architecture notes
```

## Escalation Thresholds

| Condition | Action |
|-----------|--------|
| Confidence >= 0.85 | Auto-resolve |
| Confidence 0.60–0.85 | Human reviews agent draft |
| Confidence < 0.60 | Urgent escalation |
| Amount > $100,000 | Manager approval required |
| Amount > $50,000 + high-risk intent | Senior review |
| Fraud/compromise indicators | Immediate escalation |
| Account status != Active | Manual verification |

## Observability

With `LANGCHAIN_TRACING_V2=true` and a LangSmith API key, all agent runs are traced including:
- Classification inputs/outputs
- RAG retrieval chunks and relevance
- Tool calls and responses
- Escalation decision paths
- End-to-end latency and token usage

## License

MIT
