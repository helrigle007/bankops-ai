"""
BankOps AI — Agentic Banking Customer Service Platform

Entry point for processing customer requests through the
multi-agent pipeline with classification, RAG, tool calling,
and human-in-the-loop escalation.

Usage:
    python main.py                    # Run interactive demo
    python main.py --eval             # Run evaluation suite
    python main.py --ingest           # Build/rebuild knowledge base
    python main.py --request "..."    # Process a single request
"""

import argparse
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

load_dotenv()
console = Console()


def run_interactive_demo():
    """Run an interactive demo processing sample banking requests."""
    from agents.supervisor import process_request

    sample_requests = [
        "I need to send a wire transfer of $75,000 from account ACC-10042 to our vendor at Chase Bank. Routing number 021000021, account 483291056, beneficiary Apex Supply Co.",
        "There's a $3,200 charge on account ACC-10042 from March 5th that I did not authorize. I think someone may have compromised our account.",
        "What's the current balance on our money market account ACC-10156?",
        "What are the OFAC screening requirements for international wire transfers?",
        "We need to add our new CFO David Park as an authorized signer on account ACC-10089.",
    ]

    console.print(Panel(
        "[bold]BankOps AI — Agentic Banking Customer Service Platform[/bold]\n\n"
        "Processing sample customer requests through the multi-agent pipeline.\n"
        "Each request flows through: Classification → Escalation Check → Agent Routing → Response",
        title="Demo Mode",
        border_style="blue",
    ))

    for i, request in enumerate(sample_requests, 1):
        console.print(f"\n{'='*70}")
        console.print(f"[bold cyan]Request {i}:[/bold cyan] {request}\n")

        try:
            result = process_request(request)
            console.print(f"[bold green]Response:[/bold green] {result['response']}\n")
            console.print("[dim]Metadata:[/dim]")
            console.print(JSON(json.dumps(result["metadata"], indent=2, default=str)))
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

        console.print()


def run_single_request(request: str):
    """Process a single customer request."""
    from agents.supervisor import process_request

    console.print(f"\n[bold cyan]Processing:[/bold cyan] {request}\n")
    result = process_request(request)
    console.print(f"[bold green]Response:[/bold green] {result['response']}\n")
    console.print(JSON(json.dumps(result, indent=2, default=str)))


def run_eval():
    """Run the evaluation suite."""
    from evals.eval_runner import run_full_eval
    run_full_eval(verbose=True)


def run_ingest():
    """Build or rebuild the knowledge base."""
    from knowledge.ingest import load_policies, chunk_documents, build_vectorstore

    docs = load_policies()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    console.print("[bold green]Knowledge base built successfully.[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BankOps AI — Agentic Banking Platform")
    parser.add_argument("--eval", action="store_true", help="Run evaluation suite")
    parser.add_argument("--ingest", action="store_true", help="Build/rebuild knowledge base")
    parser.add_argument("--request", type=str, help="Process a single request")

    args = parser.parse_args()

    if args.eval:
        run_eval()
    elif args.ingest:
        run_ingest()
    elif args.request:
        run_single_request(args.request)
    else:
        run_interactive_demo()
