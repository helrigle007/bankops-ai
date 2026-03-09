# BankOps AI — Project Context

## What This Is

A portfolio project demonstrating agentic AI for commercial banking customer service. Built to align with a specific AI Developer / Agentic AI Engineer job description targeting regulated financial services.

This is a **demonstration platform** — tools are simulated, policy docs are samples. The goal is to show the full lifecycle: classification → routing → RAG grounding → tool calling → human-in-the-loop escalation → evaluation.

## Tech Stack

- **Python 3.11+**
- **LangChain** / **LangGraph** — agent orchestration and graph-based workflows
- **LangSmith** — tracing and eval tracking (free developer plan, 5k traces/month)
- **ChromaDB** — local vector storage for RAG
- **Anthropic Claude** — claude-sonnet-4-6 for agents
- **HuggingFace** — all-MiniLM-L6-v2 for local embeddings (no API key needed)
- **Pydantic** — structured outputs and validation
- **Rich** — terminal formatting

## Project Structure

```
bankops-ai/
├── main.py                    # Entry point — demo, eval, ingest, single request
├── agents/
│   ├── supervisor.py          # LangGraph supervisor (planner/router pattern)
│   ├── classification.py      # Intent classification + entity extraction (structured output)
│   ├── policy_agent.py        # RAG-grounded policy Q&A agent
│   └── escalation.py          # Human-in-the-loop escalation engine (deterministic, not LLM)
├── tools/
│   └── banking_tools.py       # Simulated banking tools (account lookup, wire, case mgmt)
├── knowledge/
│   ├── policies/              # Sample banking policy markdown docs (wire, dispute, account)
│   └── ingest.py              # Ingestion pipeline — load, chunk, embed, store in ChromaDB
├── evals/
│   ├── golden_dataset.json    # 10 test cases with expected intents, entities, escalation paths
│   └── eval_runner.py         # Evaluation framework with scoring and Rich output
└── docs/
    └── architecture.md        # Design decisions and production considerations
```

## How the Pipeline Works

1. **Classify** (`agents/classification.py`) — LLM with structured output (Pydantic) classifies the request into one of 6 intent categories. Extracts account IDs, amounts, case numbers, urgency, and flags for immediate escalation.

2. **Escalation check** (`agents/escalation.py`) — Deterministic Python rule engine (not LLM) evaluates confidence scores, dollar amounts, account status, and risk indicators. Returns one of: `auto_resolve`, `human_review`, `urgent_escalation`, `manager_approval`.

3. **Route** (`agents/supervisor.py`) — LangGraph conditional edge routes to either the appropriate sub-agent or the human escalation queue.

4. **Sub-agents**:
   - `wire_transfer`, `dispute`, `account_inquiry`, `account_maintenance` → Tool-calling ReAct agent with banking tools
   - `compliance_question` → RAG chain over policy documents via ChromaDB
   - `general_inquiry` → Fallback with escalation

5. **Output** — Structured JSON with response, metadata (intent, confidence, tools called, escalation path), and audit trail.

## Key Design Decisions

- **Escalation is deterministic, not LLM-driven.** The escalation engine is pure Python with configurable thresholds. This keeps it auditable for compliance and testable without LLM variance. Thresholds live in `agents/escalation.py` as constants.

- **Classification uses structured output.** `with_structured_output()` on the LLM returns a Pydantic model, not free text. This gives us type-safe confidence scores and entities that the rest of the pipeline can rely on.

- **RAG chunking is markdown-aware.** The text splitter uses markdown separators (`## `, `### `, `- `) to preserve policy document structure. Banking policies are hierarchical — sections shouldn't get split mid-thought.

- **Tools are simulated with realistic data.** `tools/banking_tools.py` has hardcoded account and case data. In production these would be API calls to CRM, core banking, and case management. The tool signatures and return shapes are designed to be realistic.

## Running the Project

```bash
pip install -r requirements.txt
cp .env.example .env
# Add ANTHROPIC_API_KEY and LANGCHAIN_API_KEY to .env

python main.py --ingest     # Build ChromaDB vectorstore from policy docs
python main.py              # Interactive demo with 5 sample requests
python main.py --eval       # Run golden dataset evaluation
python main.py --request "Check balance on ACC-10042"
```

## Future Improvements
- Add LangSmith evaluation integration (log eval runs as datasets/experiments)
- Add a Streamlit or Gradio UI for interactive demo
- Add more policy documents (ACH, loan servicing, KYC)
- Add streaming support for the supervisor graph
- Add unit tests for the escalation engine (it's deterministic, easy to test)
- Add cost tracking per request (token counts from LangSmith traces)
- Add a Docker setup for reproducibility

## Do not change
- The escalation thresholds in `agents/escalation.py` are intentional — they map to the JD's description of confidence/risk-based HITL routing
- The simulated tool data in `tools/banking_tools.py` is referenced by the golden dataset eval cases — if you change account IDs or amounts, update the evals too
- The policy docs in `knowledge/policies/` are written to be realistic for commercial banking — keep them consistent if adding new ones

## Context: Why This Project Exists

This is a portfolio piece for a specific job application. The target role is AI Developer / Agentic AI Engineer at a commercial bank. The JD emphasizes:

- LLM-powered agents for customer service email/case intake
- RAG for policy and procedure grounding
- Tool calling to downstream systems (CRM, core banking, case management)
- Human-in-the-loop workflows based on confidence/risk thresholds
- Evaluation frameworks (golden datasets, scenario tests, scoring)
- Production readiness (observability, CI/CD for prompts, compliance)

Every module in this project maps directly to a JD responsibility. The README, architecture doc, and code comments are written with this audience in mind.

## Owner

Jason Helrigle — jason.helrigle@gmail.com — github.com/helrigle007
