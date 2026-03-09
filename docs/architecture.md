# Architecture Notes

## Design Decisions

### Why LangGraph over vanilla LangChain agents
The supervisor pattern requires conditional routing (escalation vs. auto-resolve) that's cleaner to express as a graph than as nested agent calls. LangGraph also gives us state management across nodes and makes the pipeline testable at each step.

### Why structured outputs for classification
Using Pydantic models with `with_structured_output()` instead of free-text classification gives us:
- Type-safe confidence scores we can threshold against
- Reliable entity extraction (account IDs, amounts)
- Deterministic escalation logic downstream
- Clean integration with the eval framework

### Why a separate escalation engine
Keeping escalation logic out of the LLM and in deterministic Python code means:
- Thresholds are auditable and configurable without prompt changes
- Compliance can review and approve escalation rules independently
- No risk of the LLM deciding to skip escalation on a high-risk request
- Rules can be A/B tested without affecting agent behavior

### RAG chunking strategy
Using `RecursiveCharacterTextSplitter` with markdown-aware separators (`## `, `### `, `- `) preserves the policy document structure. This matters because banking policies are hierarchical — a chunk about "Authorization Thresholds" should stay grouped, not split across two retrieval hits.

## Production Considerations

This is a demonstration platform. In a production deployment, the following would change:

- **Tool integrations**: Simulated tools would connect to real CRM, core banking, and case management APIs via secure service accounts
- **Authentication**: All tool calls would go through an API gateway with mTLS and OAuth2 token validation
- **PII handling**: Customer data would be masked/tokenized before reaching the LLM, with a detokenization step on the output side
- **Audit logging**: Every agent decision would be logged to an immutable audit store with correlation IDs for regulatory traceability
- **Model fallbacks**: Primary model failures would fall back to a secondary model with degraded but functional service
- **Rate limiting**: Token budgets per request to prevent runaway agent loops
- **Secrets management**: API keys managed through AWS Secrets Manager or HashiCorp Vault, never in environment variables
- **Model choice**: LLM calls use Anthropic Claude (claude-sonnet-4-6) for strong structured output and tool-calling support. Embeddings use HuggingFace all-MiniLM-L6-v2 running locally — no external embedding API dependency, reducing latency and cost for the retrieval pipeline
