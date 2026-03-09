"""
Evaluation framework for the BankOps AI agent pipeline.

Runs golden dataset test cases through the classification and escalation
pipeline and scores accuracy across intent, confidence, entity extraction,
and escalation routing.

Designed to integrate with LangSmith for tracing and tracking eval runs.
"""

import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from agents.classification import classify_request
from agents.escalation import evaluate_escalation


console = Console()
GOLDEN_DATASET = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    """Load golden test cases from JSON."""
    with open(GOLDEN_DATASET, "r") as f:
        return json.load(f)


def evaluate_single(test_case: dict) -> dict:
    """
    Run a single test case through classification + escalation
    and score against expected results.
    """
    request = test_case["request"]

    # Run classification
    cls_result = classify_request(request)

    # Run escalation evaluation
    esc_result = evaluate_escalation(
        intent=cls_result.intent.value,
        confidence=cls_result.confidence,
        amount=cls_result.amount or 0.0,
        requires_escalation=cls_result.requires_escalation,
        escalation_reason=cls_result.escalation_reason,
    )

    # Score results
    scores = {
        "intent_correct": cls_result.intent.value == test_case["expected_intent"],
        "confidence_above_min": cls_result.confidence >= test_case.get("expected_min_confidence", 0),
        "escalation_correct": esc_result.level.value == test_case["expected_escalation"],
    }

    # Entity extraction scoring
    expected_entities = test_case.get("expected_entities", {})
    entity_scores = {}
    if "account_id" in expected_entities:
        entity_scores["account_id"] = (
            cls_result.account_id is not None
            and cls_result.account_id.upper() == expected_entities["account_id"].upper()
        )
    if "amount" in expected_entities:
        entity_scores["amount"] = (
            cls_result.amount is not None
            and abs(cls_result.amount - expected_entities["amount"]) < 0.01
        )
    if "case_id" in expected_entities:
        entity_scores["case_id"] = (
            cls_result.case_id is not None
            and cls_result.case_id.upper() == expected_entities["case_id"].upper()
        )
    scores["entities"] = entity_scores
    scores["all_entities_correct"] = all(entity_scores.values()) if entity_scores else True

    return {
        "test_id": test_case["id"],
        "request": request[:80] + "..." if len(request) > 80 else request,
        "classification": cls_result.model_dump(),
        "escalation": esc_result.model_dump(),
        "scores": scores,
        "passed": all([
            scores["intent_correct"],
            scores["escalation_correct"],
            scores["all_entities_correct"],
        ]),
    }


def run_full_eval(verbose: bool = True) -> dict:
    """
    Run all golden dataset test cases and produce a summary report.
    """
    dataset = load_golden_dataset()
    results = []

    console.print(f"\n[bold]Running {len(dataset)} evaluation cases...[/bold]\n")

    for case in dataset:
        result = evaluate_single(case)
        results.append(result)

        if verbose:
            status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
            console.print(f"  {result['test_id']}: {status} — {result['request']}")

    # Aggregate scores
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    intent_correct = sum(1 for r in results if r["scores"]["intent_correct"])
    escalation_correct = sum(1 for r in results if r["scores"]["escalation_correct"])
    entity_correct = sum(1 for r in results if r["scores"]["all_entities_correct"])

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0,
        "intent_accuracy": intent_correct / total if total > 0 else 0,
        "escalation_accuracy": escalation_correct / total if total > 0 else 0,
        "entity_accuracy": entity_correct / total if total > 0 else 0,
        "results": results,
    }

    if verbose:
        _print_summary_table(summary)

    return summary


def _print_summary_table(summary: dict):
    """Print a formatted summary table."""
    console.print(f"\n{'='*60}")
    console.print("[bold]EVALUATION SUMMARY[/bold]")
    console.print(f"{'='*60}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Pct", justify="right")

    table.add_row(
        "Overall Pass Rate",
        f"{summary['passed']}/{summary['total_cases']}",
        f"{summary['pass_rate']:.0%}",
    )
    table.add_row(
        "Intent Classification",
        f"{int(summary['intent_accuracy'] * summary['total_cases'])}/{summary['total_cases']}",
        f"{summary['intent_accuracy']:.0%}",
    )
    table.add_row(
        "Escalation Routing",
        f"{int(summary['escalation_accuracy'] * summary['total_cases'])}/{summary['total_cases']}",
        f"{summary['escalation_accuracy']:.0%}",
    )
    table.add_row(
        "Entity Extraction",
        f"{int(summary['entity_accuracy'] * summary['total_cases'])}/{summary['total_cases']}",
        f"{summary['entity_accuracy']:.0%}",
    )

    console.print(table)

    # Show failures
    failures = [r for r in summary["results"] if not r["passed"]]
    if failures:
        console.print(f"\n[bold red]Failed Cases ({len(failures)}):[/bold red]")
        for f in failures:
            scores = f["scores"]
            reasons = []
            if not scores["intent_correct"]:
                expected = next(c for c in summary["results"] if c["test_id"] == f["test_id"])
                reasons.append(f"intent: got {f['classification']['intent']}")
            if not scores["escalation_correct"]:
                reasons.append(f"escalation: got {f['escalation']['level']}")
            if not scores["all_entities_correct"]:
                reasons.append(f"entities: {scores['entities']}")
            console.print(f"  {f['test_id']}: {', '.join(reasons)}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    run_full_eval(verbose=True)
