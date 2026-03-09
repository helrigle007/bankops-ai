"""
Simulated banking tools for agent tool-calling demonstrations.

In production these would integrate with CRM, core banking,
case management, and payment systems via secure APIs.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
import random


# --- Simulated data stores ---

ACCOUNTS = {
    "ACC-10042": {
        "name": "Meridian Manufacturing LLC",
        "type": "Commercial Checking",
        "balance": 284_750.33,
        "status": "Active",
        "signers": ["James Thornton", "Patricia Vega"],
        "wire_limit": 100_000,
        "opened": "2019-03-15",
    },
    "ACC-10089": {
        "name": "Coastal Freight Partners",
        "type": "Commercial Checking",
        "balance": 1_203_418.90,
        "status": "Active",
        "signers": ["Robert Kim", "Linda Morales", "David Chen"],
        "wire_limit": 500_000,
        "opened": "2017-08-22",
    },
    "ACC-10156": {
        "name": "Apex Digital Solutions Inc",
        "type": "Money Market",
        "balance": 52_100.00,
        "status": "Under Review",
        "signers": ["Sarah Mitchell"],
        "wire_limit": 25_000,
        "opened": "2023-01-10",
    },
}

CASES = {
    "CASE-8812": {
        "account": "ACC-10042",
        "type": "Unauthorized Transaction",
        "amount": 3_200.00,
        "status": "Under Investigation",
        "created": "2026-03-05",
        "assigned_to": "Disputes Team",
    },
    "CASE-8830": {
        "account": "ACC-10089",
        "type": "Wire Transfer Failure",
        "amount": 45_000.00,
        "status": "Pending",
        "created": "2026-03-08",
        "assigned_to": "Wire Operations",
    },
}


# --- Tool definitions ---

@tool
def lookup_account(account_id: str) -> dict:
    """Look up a commercial banking account by account ID. Returns account details including name, type, balance, status, authorized signers, and wire limit."""
    account = ACCOUNTS.get(account_id.upper())
    if not account:
        return {"error": f"Account {account_id} not found", "suggestion": "Verify the account number and try again"}
    return {"account_id": account_id.upper(), **account}


@tool
def get_account_transactions(account_id: str, days: int = 30) -> dict:
    """Retrieve recent transactions for a commercial banking account. Returns the last N days of transaction history."""
    account = ACCOUNTS.get(account_id.upper())
    if not account:
        return {"error": f"Account {account_id} not found"}

    # Simulated transaction history
    transactions = [
        {"date": "2026-03-08", "description": "Wire Transfer Out - Vendor Payment", "amount": -15_420.00, "balance": account["balance"]},
        {"date": "2026-03-07", "description": "ACH Deposit - Client Payment", "amount": 28_500.00, "balance": account["balance"] + 15_420.00},
        {"date": "2026-03-06", "description": "Service Charge - Wire Fee", "amount": -25.00, "balance": account["balance"] + 15_420.00 - 28_500.00},
        {"date": "2026-03-05", "description": "Check Deposit #4521", "amount": 12_300.00, "balance": account["balance"] + 15_420.00 - 28_500.00 + 25.00},
        {"date": "2026-03-03", "description": "ACH Debit - Payroll", "amount": -42_680.50, "balance": account["balance"] + 15_420.00 - 28_500.00 + 25.00 - 12_300.00},
    ]
    return {"account_id": account_id.upper(), "period_days": days, "transactions": transactions}


@tool
def lookup_case(case_id: str) -> dict:
    """Look up an existing dispute or service case by case ID. Returns case details including status, type, and assigned team."""
    case = CASES.get(case_id.upper())
    if not case:
        return {"error": f"Case {case_id} not found", "suggestion": "Verify the case number format (CASE-XXXX)"}
    return {"case_id": case_id.upper(), **case}


@tool
def create_case(
    account_id: str,
    case_type: str,
    description: str,
    amount: float = 0.0,
    priority: str = "normal",
) -> dict:
    """Create a new service case in the case management system. Use for disputes, service requests, and escalations."""
    case_id = f"CASE-{random.randint(9000, 9999)}"
    case = {
        "case_id": case_id,
        "account_id": account_id.upper(),
        "type": case_type,
        "description": description,
        "amount": amount,
        "priority": priority,
        "status": "Open",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "assigned_to": _route_case(case_type, amount),
    }
    return case


@tool
def initiate_wire_transfer(
    from_account: str,
    to_account: str,
    to_bank_routing: str,
    amount: float,
    beneficiary_name: str,
    purpose: str = "",
) -> dict:
    """Initiate a wire transfer from a commercial banking account. Validates authorization thresholds and compliance checks before submission."""
    account = ACCOUNTS.get(from_account.upper())
    if not account:
        return {"error": f"Account {from_account} not found"}

    if account["status"] != "Active":
        return {"error": f"Account is {account['status']} — wire transfers not permitted"}

    if amount > account["balance"]:
        return {"error": "Insufficient funds", "available_balance": account["balance"]}

    # Determine authorization level needed
    auth_level = _get_wire_auth_level(amount)

    # Check if amount exceeds account wire limit
    needs_override = amount > account["wire_limit"]

    return {
        "wire_id": f"WR-{random.randint(100000, 999999)}",
        "status": "Pending Approval" if needs_override else "Submitted",
        "from_account": from_account.upper(),
        "beneficiary": beneficiary_name,
        "amount": amount,
        "authorization_required": auth_level,
        "exceeds_wire_limit": needs_override,
        "compliance_screening": "Passed" if random.random() > 0.1 else "OFAC Review Required",
        "estimated_processing": "Same day" if datetime.now().hour < 16 else "Next business day",
    }


@tool
def check_compliance_status(account_id: str) -> dict:
    """Check compliance and regulatory status for a commercial account. Returns KYC, BSA/AML, and beneficial ownership status."""
    account = ACCOUNTS.get(account_id.upper())
    if not account:
        return {"error": f"Account {account_id} not found"}

    return {
        "account_id": account_id.upper(),
        "kyc_status": "Current",
        "kyc_last_review": "2025-09-15",
        "kyc_next_review": "2026-09-15",
        "beneficial_ownership": "On file — verified",
        "bsa_aml_alerts": 0,
        "ofac_screening": "Clear",
        "risk_rating": "Low",
        "suspicious_activity": False,
    }


# --- Helper functions ---

def _get_wire_auth_level(amount: float) -> str:
    if amount < 10_000:
        return "Single signer"
    elif amount <= 100_000:
        return "Dual authorization"
    else:
        return "Dual authorization + manager approval"


def _route_case(case_type: str, amount: float) -> str:
    routing = {
        "Unauthorized Transaction": "Fraud Operations" if amount > 0 else "Disputes Team",
        "Billing Error": "Disputes Team",
        "Service Charge Dispute": "Customer Service",
        "Wire Transfer Failure": "Wire Operations",
        "Fraud Claim": "Fraud Operations",
        "Account Maintenance": "Account Services",
    }
    team = routing.get(case_type, "General Service")
    if amount > 50_000:
        team = f"{team} → Escalated to Operations Manager"
    return team


# Collect all tools for agent binding
all_tools = [
    lookup_account,
    get_account_transactions,
    lookup_case,
    create_case,
    initiate_wire_transfer,
    check_compliance_status,
]
