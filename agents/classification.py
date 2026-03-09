"""
Intent classification and entity extraction for inbound banking requests.

Classifies customer service emails/messages into intent categories
and extracts structured entities (account numbers, amounts, dates)
for downstream agent routing.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class IntentCategory(str, Enum):
    WIRE_TRANSFER = "wire_transfer"
    DISPUTE = "dispute"
    ACCOUNT_INQUIRY = "account_inquiry"
    ACCOUNT_MAINTENANCE = "account_maintenance"
    COMPLIANCE_QUESTION = "compliance_question"
    GENERAL_INQUIRY = "general_inquiry"


class ClassificationResult(BaseModel):
    """Structured output from intent classification."""
    intent: IntentCategory = Field(description="Primary intent category of the request")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    account_id: Optional[str] = Field(default=None, description="Extracted account number if present")
    amount: Optional[float] = Field(default=None, description="Extracted monetary amount if present")
    case_id: Optional[str] = Field(default=None, description="Extracted case/reference number if present")
    urgency: str = Field(default="normal", description="Assessed urgency: low, normal, high, critical")
    summary: str = Field(description="One-sentence summary of the customer request")
    requires_escalation: bool = Field(default=False, description="Whether this request should bypass normal routing and escalate immediately")
    escalation_reason: Optional[str] = Field(default=None, description="Reason for escalation if flagged")


CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for a commercial banking customer service platform.

Analyze the incoming customer request and extract:
1. The primary intent category
2. Your confidence level (0.0 to 1.0)
3. Any structured entities (account numbers, amounts, case IDs)
4. Urgency assessment
5. Whether this needs immediate escalation

Intent categories:
- wire_transfer: Wire initiation, status, recall, or wire-related questions
- dispute: Unauthorized transactions, billing errors, chargebacks, fraud claims, account compromise, unauthorized access or changes
- account_inquiry: Balance checks, transaction history, account status
- account_maintenance: Routine signer changes, address updates, account closure, product changes (NOT when fraud or compromise is involved)
- compliance_question: Policy questions, regulatory requirements, KYC/BSA inquiries
- general_inquiry: Anything that doesn't fit above categories

IMPORTANT: If a request mentions account compromise, unauthorized changes, or fraud — even if it also mentions signers or account details — classify as "dispute", not "account_maintenance".

Escalation triggers (set requires_escalation=True):
- Fraud or unauthorized access mentioned
- Amounts over $100,000
- Legal or regulatory threats
- Time-sensitive wire recalls
- Account compromise indicators

Be precise with confidence scores:
- 0.9+ : Clear, unambiguous intent
- 0.7-0.9 : Likely intent but some ambiguity
- 0.5-0.7 : Unclear, multiple possible intents
- Below 0.5 : Cannot determine intent reliably
"""),
    ("human", "{request}"),
])


def get_classifier(model: str = "claude-sonnet-4-6"):
    """Return a classification chain with structured output."""
    llm = ChatAnthropic(model=model, temperature=0)
    structured_llm = llm.with_structured_output(ClassificationResult)
    chain = CLASSIFICATION_PROMPT | structured_llm
    return chain


def classify_request(request: str, model: str = "claude-sonnet-4-6") -> ClassificationResult:
    """Classify a single customer request and return structured result."""
    chain = get_classifier(model)
    return chain.invoke({"request": request})


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_requests = [
        "Hi, I need to send a wire transfer of $75,000 to our vendor's account at Chase. Account ACC-10042.",
        "I'm seeing a charge for $3,200 on account ACC-10042 that I did not authorize. This needs to be investigated immediately.",
        "Can you tell me the current balance on our money market account ACC-10156?",
        "We need to add David Park as an authorized signer on account ACC-10089. He's our new CFO.",
        "What are the OFAC screening requirements for international wire transfers?",
    ]

    for req in test_requests:
        result = classify_request(req)
        print(f"\nRequest: {req[:60]}...")
        print(f"  Intent: {result.intent.value} (confidence: {result.confidence})")
        print(f"  Urgency: {result.urgency} | Escalate: {result.requires_escalation}")
        print(f"  Summary: {result.summary}")
