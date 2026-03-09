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
from rich.table import Table
from rich.markdown import Markdown
from rich.rule import Rule
from rich.columns import Columns
from rich.text import Text
from rich.status import Status

load_dotenv()
console = Console()


ESCALATION_STYLES = {
    "auto_resolve": ("green", "AUTO-RESOLVE"),
    "human_review": ("yellow", "HUMAN REVIEW"),
    "urgent_escalation": ("red", "URGENT ESCALATION"),
    "manager_approval": ("bright_red", "MANAGER APPROVAL"),
}


def _format_result(result: dict):
    """Format a pipeline result with Rich panels and tables."""
    meta = result.get("metadata", {})
    escalation_level = meta.get("escalation_level", "auto_resolve")
    esc_color, esc_label = ESCALATION_STYLES.get(escalation_level, ("white", escalation_level))

    # Pipeline metadata table
    pipeline_table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=True,
    )
    pipeline_table.add_column("Label", style="dim", width=20)
    pipeline_table.add_column("Value")

    raw_intent = str(meta.get("intent", "unknown"))
    intent_display = raw_intent.split(".")[-1].lower() if "." in raw_intent else raw_intent
    pipeline_table.add_row("Intent", f"[bold]{intent_display}[/bold]")
    pipeline_table.add_row("Confidence", f"{meta.get('confidence', 0):.0%}")
    pipeline_table.add_row("Urgency", meta.get("urgency", "normal"))
    pipeline_table.add_row("Escalation", f"[bold {esc_color}]{esc_label}[/bold {esc_color}]")

    tools = meta.get("tools_called", [])
    if tools:
        pipeline_table.add_row("Tools Called", ", ".join(tools))

    console.print(Panel(
        pipeline_table,
        title="[bold]Pipeline Output[/bold]",
        border_style="dim",
        padding=(0, 0),
    ))

    # Response panel — color border by escalation level
    response_text = result.get("response", "No response generated.")
    console.print(Panel(
        Markdown(response_text),
        title=f"[bold {esc_color}]Agent Response[/bold {esc_color}]",
        border_style=esc_color,
        padding=(1, 2),
    ))

    # Escalation details if present
    if result.get("escalation_details"):
        esc = result["escalation_details"]
        esc_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        esc_table.add_column("Label", style="dim", width=20)
        esc_table.add_column("Value")
        esc_table.add_row("Queue", esc.get("assigned_queue", ""))
        esc_table.add_row("SLA", f"{esc.get('sla_hours', '')} hours")
        esc_table.add_row("Reason", esc.get("reason", ""))
        rules = esc.get("triggered_rules", [])
        if rules:
            esc_table.add_row("Triggered Rules", "\n".join(f"- {r}" for r in rules))

        console.print(Panel(
            esc_table,
            title="[bold red]Escalation Details[/bold red]",
            border_style="red",
            padding=(0, 0),
        ))


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

    console.print()
    console.print(Panel(
        "[bold white]BankOps AI[/bold white]\n"
        "[dim]Agentic Banking Customer Service Platform[/dim]\n\n"
        "Processing sample customer requests through the multi-agent pipeline.\n"
        "Each request flows through:\n\n"
        "  [cyan]Classification[/cyan] → [yellow]Escalation Check[/yellow] → [green]Agent Routing[/green] → [bold]Response[/bold]",
        title="[bold blue] Demo Mode [/bold blue]",
        border_style="blue",
        padding=(1, 3),
    ))

    for i, request in enumerate(sample_requests, 1):
        try:
            console.print("\n\n")
            console.print(Rule(f"[bold cyan] Request {i} [/bold cyan]", style="cyan"))
            console.print()
            console.print(Panel(
                request,
                title="[bold]Customer Request[/bold]",
                border_style="cyan",
                padding=(1, 2),
            ))
            with console.status(
                f"[bold cyan]Processing request {i}/{len(sample_requests)}...[/bold cyan]",
                spinner="dots",
            ):
                result = process_request(request)
            _format_result(result)
        except Exception as e:
            console.print(f"\n[bold red]Error processing request {i}:[/bold red] {e}")

    console.print()
    console.print(Rule("[bold blue] Demo Complete [/bold blue]", style="blue"))
    console.print()


def run_single_request(request: str):
    """Process a single customer request."""
    from agents.supervisor import process_request

    console.print("\n\n")
    console.print(Rule("[bold cyan] Request [/bold cyan]", style="cyan"))
    console.print()
    console.print(Panel(
        request,
        title="[bold]Customer Request[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))
    with console.status(
        "[bold cyan]Processing request...[/bold cyan]",
        spinner="dots",
    ):
        result = process_request(request)
    _format_result(result)
    console.print()


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
