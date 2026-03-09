"""
Human-in-the-loop escalation engine.

Evaluates agent responses against confidence thresholds and risk criteria
to determine whether to auto-resolve or route to human review.
Implements the HITL patterns described in production agentic AI systems.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class EscalationLevel(str, Enum):
    AUTO_RESOLVE = "auto_resolve"
    HUMAN_REVIEW = "human_review"
    URGENT_ESCALATION = "urgent_escalation"
    MANAGER_APPROVAL = "manager_approval"


class EscalationDecision(BaseModel):
    """Result of the escalation evaluation."""
    level: EscalationLevel
    reason: str
    triggered_rules: list[str] = Field(default_factory=list)
    recommended_action: str
    assigned_queue: str
    sla_hours: int
    context_for_reviewer: Optional[str] = None


# --- Configurable thresholds ---

CONFIDENCE_THRESHOLDS = {
    "auto_resolve": 0.85,       # Above this: agent handles it
    "human_review": 0.60,       # Between review and auto: human reviews agent's draft
    "urgent_escalation": 0.40,  # Below this: urgent human intervention
}

AMOUNT_THRESHOLDS = {
    "auto_resolve_max": 10_000,
    "review_required": 50_000,
    "manager_required": 100_000,
}

HIGH_RISK_INTENTS = ["dispute", "wire_transfer"]
ALWAYS_ESCALATE_INTENTS = ["fraud_claim"]


def evaluate_escalation(
    intent: str,
    confidence: float,
    amount: float = 0.0,
    account_status: str = "Active",
    requires_escalation: bool = False,
    escalation_reason: str = None,
    agent_response: str = None,
) -> EscalationDecision:
    """
    Evaluate whether a request should be auto-resolved or escalated.

    Applies a layered rule engine:
    1. Hard escalation rules (fraud, account flags)
    2. Amount-based thresholds
    3. Confidence-based routing
    4. Intent-specific overrides
    """
    triggered_rules = []

    # --- Layer 1: Hard escalation rules ---
    if requires_escalation:
        triggered_rules.append(f"Classification flagged for escalation: {escalation_reason}")
        return EscalationDecision(
            level=EscalationLevel.URGENT_ESCALATION,
            reason=f"Auto-escalation trigger: {escalation_reason}",
            triggered_rules=triggered_rules,
            recommended_action="Route to senior agent immediately",
            assigned_queue="Priority Escalations",
            sla_hours=1,
            context_for_reviewer=agent_response,
        )

    if account_status != "Active":
        triggered_rules.append(f"Account status: {account_status}")
        return EscalationDecision(
            level=EscalationLevel.HUMAN_REVIEW,
            reason=f"Account is {account_status} — requires manual verification",
            triggered_rules=triggered_rules,
            recommended_action="Verify account standing before processing",
            assigned_queue="Account Review",
            sla_hours=4,
            context_for_reviewer=agent_response,
        )

    # --- Layer 2: Amount-based thresholds ---
    if amount > AMOUNT_THRESHOLDS["manager_required"]:
        triggered_rules.append(f"Amount ${amount:,.2f} exceeds manager threshold (${AMOUNT_THRESHOLDS['manager_required']:,})")
        return EscalationDecision(
            level=EscalationLevel.MANAGER_APPROVAL,
            reason=f"High-value request: ${amount:,.2f} requires manager sign-off",
            triggered_rules=triggered_rules,
            recommended_action="Route to Operations Manager for approval",
            assigned_queue="Manager Approval Queue",
            sla_hours=2,
            context_for_reviewer=agent_response,
        )

    if amount > AMOUNT_THRESHOLDS["review_required"]:
        triggered_rules.append(f"Amount ${amount:,.2f} exceeds review threshold (${AMOUNT_THRESHOLDS['review_required']:,})")

    # --- Layer 3: Confidence-based routing ---
    if confidence >= CONFIDENCE_THRESHOLDS["auto_resolve"]:
        # High confidence — check if amount or intent overrides
        if amount > AMOUNT_THRESHOLDS["review_required"] and intent in HIGH_RISK_INTENTS:
            triggered_rules.append(f"High-risk intent '{intent}' with elevated amount")
            return EscalationDecision(
                level=EscalationLevel.HUMAN_REVIEW,
                reason="High confidence but elevated risk profile",
                triggered_rules=triggered_rules,
                recommended_action="Review agent draft response before sending",
                assigned_queue="Senior Review",
                sla_hours=4,
                context_for_reviewer=agent_response,
            )

        return EscalationDecision(
            level=EscalationLevel.AUTO_RESOLVE,
            reason=f"Confidence {confidence:.2f} above threshold — auto-resolving",
            triggered_rules=triggered_rules or ["No escalation rules triggered"],
            recommended_action="Send agent response to customer",
            assigned_queue="Resolved",
            sla_hours=0,
        )

    elif confidence >= CONFIDENCE_THRESHOLDS["human_review"]:
        triggered_rules.append(f"Confidence {confidence:.2f} in review range ({CONFIDENCE_THRESHOLDS['human_review']}-{CONFIDENCE_THRESHOLDS['auto_resolve']})")
        return EscalationDecision(
            level=EscalationLevel.HUMAN_REVIEW,
            reason="Moderate confidence — human should review before responding",
            triggered_rules=triggered_rules,
            recommended_action="Agent drafted a response — review and approve or edit",
            assigned_queue="Human Review",
            sla_hours=8,
            context_for_reviewer=agent_response,
        )

    else:
        triggered_rules.append(f"Confidence {confidence:.2f} below minimum threshold ({CONFIDENCE_THRESHOLDS['human_review']})")
        return EscalationDecision(
            level=EscalationLevel.URGENT_ESCALATION,
            reason="Low confidence — agent cannot reliably handle this request",
            triggered_rules=triggered_rules,
            recommended_action="Assign to experienced agent for manual handling",
            assigned_queue="Priority Escalations",
            sla_hours=2,
            context_for_reviewer=agent_response,
        )


if __name__ == "__main__":
    # Demo: show different escalation paths
    scenarios = [
        {"intent": "account_inquiry", "confidence": 0.95, "amount": 0, "label": "Simple balance check"},
        {"intent": "wire_transfer", "confidence": 0.88, "amount": 75_000, "label": "High-value wire"},
        {"intent": "dispute", "confidence": 0.65, "amount": 3_200, "label": "Unclear dispute"},
        {"intent": "wire_transfer", "confidence": 0.30, "amount": 250_000, "label": "Ambiguous high-value request"},
        {"intent": "dispute", "confidence": 0.92, "amount": 3_200, "requires_escalation": True, "escalation_reason": "Possible account compromise", "label": "Fraud flag"},
    ]

    for s in scenarios:
        label = s.pop("label")
        result = evaluate_escalation(**s)
        print(f"\n{'='*60}")
        print(f"Scenario: {label}")
        print(f"  Decision: {result.level.value}")
        print(f"  Reason: {result.reason}")
        print(f"  Queue: {result.assigned_queue} (SLA: {result.sla_hours}h)")
        print(f"  Rules: {', '.join(result.triggered_rules)}")
