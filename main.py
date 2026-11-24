"""
Aegis Procure: Multi-Agent Enterprise Procurement CLI
Simple, direct terminal interface with low-overhead latency tracing.
"""

import sys
import os
from pathlib import Path

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from agents_core.guardrails import run_input_guardrail, run_output_guardrail
from agents_core.specialists import AegisMultiAgentRunner
from agents_core.tracing_hooks import TracingHook

try:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
except ImportError:
    console = None


def print_msg(text: str, style: str = ""):
    if console:
        console.print(text, style=style)
    else:
        print(text)


def process_query(runner: AegisMultiAgentRunner, tracer: TracingHook, query: str):
    """Processes a single procurement inquiry through the full multi-agent pipeline."""
    if not query.strip():
        return

    print_msg(f"\n[Aegis Procure] > {query}", style="bold white")
    tracer.start_turn()

    # Step 1: Perimeter Input Guardrail
    passed, scope_check = run_input_guardrail(query)
    tracer.log_guardrail("Perimeter Scope & Injection", passed, scope_check.rejection_reason)

    if not passed:
        print_msg(f"\n[Rejection] {scope_check.rejection_reason}\n", style="bold red")
        tracer.end_turn()
        return

    # Step 2: Multi-Agent Turn Execution with Live Handoffs
    def step_callback(event_type: str, detail: str):
        if event_type == "HANDOFF":
            tracer.log_handoff(detail)
        elif event_type == "TOOL_START":
            tracer.log_tool(detail)

    turn_result = runner.execute_turn(
        user_query=query, 
        intent_category=scope_check.intent_category,
        step_callback=step_callback
    )

    # Step 3: Output Guardrail (PII Redaction & Grounding)
    raw_answer = turn_result["response"]
    out_passed, clean_answer = run_output_guardrail(raw_answer)
    tracer.log_guardrail("Output Grounding & Privacy", out_passed)

    # Step 4: Render Final Clean Result
    print_msg("\n[Response]", style="bold green")
    if console:
        console.print(Markdown(clean_answer))
    else:
        print(clean_answer)

    tracer.end_turn()


def run_demo(runner: AegisMultiAgentRunner, tracer: TracingHook):
    """Executes the 3 representative enterprise scenarios automatically."""
    demo_queries = [
        "What is the penalty for delayed deliverables in Acme Cloud Services SLA?",
        "Show total spend grouped by vendor tier from ERP purchase orders.",
        "Compare Apex IT Solutions spend against their contractual spend cap."
    ]
    print_msg("=== Running Aegis Procure Automated Verification Demo ===", style="bold yellow")
    for q in demo_queries:
        process_query(runner, tracer, q)


def main():
    runner = AegisMultiAgentRunner()
    tracer = TracingHook()

    # Check CLI arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ["--demo", "-d", "demo"]:
            run_demo(runner, tracer)
            return
        # One-shot query execution
        full_query = " ".join(sys.argv[1:])
        process_query(runner, tracer, full_query)
        return

    # Simple interactive terminal loop
    print_msg("Aegis Procure Multi-Agent Runtime (Type 'demo' for test queries, 'exit' to quit)\n", style="bold cyan")
    
    while True:
        try:
            user_input = input("[Aegis Procure] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print_msg("Exiting Aegis Procure.", style="dim")
                break
            if user_input.lower() == "demo":
                run_demo(runner, tracer)
                continue

            process_query(runner, tracer, user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
