"""
LangGraph supervisor agent for commercial banking customer service.

Implements a planner/supervisor pattern that:
1. Classifies inbound requests
2. Routes to specialized sub-agents
3. Evaluates confidence and risk for HITL escalation
4. Produces structured responses with audit trails
"""

from typing import TypedDict, Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from agents.classification import classify_request, ClassificationResult
from agents.escalation import evaluate_escalation, EscalationLevel
from agents.policy_agent import query_policy
from tools.banking_tools import (
    lookup_account,
    get_account_transactions,
    lookup_case,
    create_case,
    initiate_wire_transfer,
    check_compliance_status,
    all_tools,
)


# --- State definition ---

class AgentState(TypedDict):
    """Shared state across the agent graph."""
    request: str
    classification: dict | None
    escalation: dict | None
    agent_response: str | None
    tools_called: list[str]
    route: str | None
    final_output: dict | None


# --- Node functions ---

def classify_node(state: AgentState) -> AgentState:
    """Classify the inbound request and extract entities."""
    result = classify_request(state["request"])
    return {
        **state,
        "classification": result.model_dump(),
        "route": result.intent.value,
    }


def check_escalation_node(state: AgentState) -> AgentState:
    """Evaluate whether the request should be auto-resolved or escalated."""
    cls = state["classification"]
    decision = evaluate_escalation(
        intent=cls["intent"],
        confidence=cls["confidence"],
        amount=cls.get("amount") or 0.0,
        requires_escalation=cls.get("requires_escalation", False),
        escalation_reason=cls.get("escalation_reason"),
    )
    return {
        **state,
        "escalation": decision.model_dump(),
    }


def route_by_escalation(state: AgentState) -> Literal["handle_request", "escalate_to_human"]:
    """Route based on escalation decision."""
    level = state["escalation"]["level"]
    if level == EscalationLevel.AUTO_RESOLVE.value:
        return "handle_request"
    return "escalate_to_human"


def handle_request_node(state: AgentState) -> AgentState:
    """
    Route to the appropriate sub-agent based on classified intent.
    Uses tool-calling agents for transactional intents
    and RAG for policy/compliance questions.
    """
    route = state["route"]
    cls = state["classification"]
    tools_called = []

    if route == "compliance_question":
        # Use RAG policy agent
        result = query_policy(state["request"])
        response = result["answer"]
        tools_called = ["policy_rag_retriever"]

    elif route in ("wire_transfer", "dispute", "account_inquiry", "account_maintenance"):
        # Use tool-calling agent
        llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
        agent = create_react_agent(llm, all_tools)

        system_context = _get_system_prompt(route)
        result = agent.invoke({
            "messages": [
                SystemMessage(content=system_context),
                HumanMessage(content=state["request"]),
            ]
        })
        response = result["messages"][-1].content
        tools_called = [
            msg.tool_calls[0]["name"]
            for msg in result["messages"]
            if hasattr(msg, "tool_calls") and msg.tool_calls
        ]

    else:
        response = (
            "I've received your request but I'm not sure how to route it. "
            "I'm escalating this to a team member who can help."
        )

    return {
        **state,
        "agent_response": response,
        "tools_called": tools_called,
    }


def escalate_to_human_node(state: AgentState) -> AgentState:
    """Build an escalation package for human review."""
    cls = state["classification"]
    esc = state["escalation"]

    escalation_package = {
        "status": "ESCALATED — Awaiting Human Review",
        "escalation_level": esc["level"],
        "assigned_queue": esc["assigned_queue"],
        "sla_hours": esc["sla_hours"],
        "reason": esc["reason"],
        "triggered_rules": esc["triggered_rules"],
        "customer_request": state["request"],
        "classification": {
            "intent": cls["intent"],
            "confidence": cls["confidence"],
            "urgency": cls["urgency"],
            "summary": cls["summary"],
        },
        "recommended_action": esc["recommended_action"],
    }

    # If we have a draft response, include it for the reviewer
    if state.get("agent_response"):
        escalation_package["draft_response"] = state["agent_response"]

    response = (
        f"This request has been escalated to {esc['assigned_queue']}.\n"
        f"Reason: {esc['reason']}\n"
        f"SLA: {esc['sla_hours']} hours\n"
        f"A team member will review and respond shortly."
    )

    return {
        **state,
        "agent_response": response,
        "final_output": escalation_package,
    }


def build_output_node(state: AgentState) -> AgentState:
    """Assemble the final structured output with full audit trail."""
    cls = state["classification"]
    esc = state["escalation"]

    output = {
        "response": state["agent_response"],
        "metadata": {
            "intent": cls["intent"],
            "confidence": cls["confidence"],
            "urgency": cls["urgency"],
            "escalation_level": esc["level"],
            "tools_called": state["tools_called"],
            "summary": cls["summary"],
        },
    }

    # Merge escalation package if it exists
    if state.get("final_output"):
        output["escalation_details"] = state["final_output"]

    return {**state, "final_output": output}


# --- System prompts per route ---

def _get_system_prompt(route: str) -> str:
    prompts = {
        "wire_transfer": (
            "You are a wire transfer specialist agent for commercial banking. "
            "Help the customer with wire transfer requests by looking up their account, "
            "validating the request, and initiating the transfer if appropriate. "
            "Always verify the account exists and has sufficient funds before proceeding. "
            "Flag any compliance concerns."
        ),
        "dispute": (
            "You are a dispute resolution agent for commercial banking. "
            "Help the customer file or check on disputes. Look up the account and any existing cases. "
            "Create new cases when needed. Be thorough in gathering details."
        ),
        "account_inquiry": (
            "You are an account service agent for commercial banking. "
            "Help the customer with account lookups, balance inquiries, and transaction history. "
            "Use the available tools to retrieve accurate account information."
        ),
        "account_maintenance": (
            "You are an account maintenance agent for commercial banking. "
            "Help with signer changes, address updates, and other account modifications. "
            "Explain required documentation and next steps clearly."
        ),
    }
    return prompts.get(route, "You are a helpful commercial banking customer service agent.")


# --- Build the graph ---

def build_graph() -> StateGraph:
    """Construct the LangGraph supervisor workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("check_escalation", check_escalation_node)
    graph.add_node("handle_request", handle_request_node)
    graph.add_node("escalate_to_human", escalate_to_human_node)
    graph.add_node("build_output", build_output_node)

    # Define edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "check_escalation")
    graph.add_conditional_edges(
        "check_escalation",
        route_by_escalation,
        {
            "handle_request": "handle_request",
            "escalate_to_human": "escalate_to_human",
        },
    )
    graph.add_edge("handle_request", "build_output")
    graph.add_edge("escalate_to_human", "build_output")
    graph.add_edge("build_output", END)

    return graph.compile()


def process_request(request: str) -> dict:
    """Process a single customer request through the full agent pipeline."""
    app = build_graph()
    initial_state = {
        "request": request,
        "classification": None,
        "escalation": None,
        "agent_response": None,
        "tools_called": [],
        "route": None,
        "final_output": None,
    }
    result = app.invoke(initial_state)
    return result["final_output"]
