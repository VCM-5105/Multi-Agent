"""
Aegis Procure: Tracing Hooks & Latency Tracker
Logs live step-by-step agent handoffs, tool invocations, and latency breakdowns.
Kept simple and non-intrusive for clean terminal display.
"""

import time
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.text import Text
    console = Console()
except ImportError:
    console = None


class TracingHook:
    """Tracks latency and logs pipeline events with minimal visual noise."""
    def __init__(self):
        self.start_time = 0.0
        self.step_start_time = 0.0
        self.events: List[Dict[str, Any]] = []

    def start_turn(self):
        self.start_time = time.perf_counter()
        self.step_start_time = self.start_time
        self.events = []

    def log_guardrail(self, name: str, passed: bool, reason: Optional[str] = None):
        elapsed_ms = (time.perf_counter() - self.step_start_time) * 1000
        self.step_start_time = time.perf_counter()
        
        status_symbol = "✔" if passed else "✘"
        badge = "PASS" if passed else "BLOCKED"
        
        if console:
            color = "green" if passed else "bold red"
            msg = f"  [{color}]{status_symbol} Guardrail: {name} [{badge}][/{color}] ({elapsed_ms:.1f}ms)"
            if reason:
                msg += f" -> [yellow]{reason}[/yellow]"
            console.print(msg)
        else:
            print(f"  {status_symbol} Guardrail: {name} [{badge}] ({elapsed_ms:.1f}ms) {reason or ''}")

    def log_handoff(self, transition: str):
        elapsed_ms = (time.perf_counter() - self.step_start_time) * 1000
        self.step_start_time = time.perf_counter()
        
        if console:
            console.print(f"  [cyan]➜ Handoff: {transition}[/cyan] ({elapsed_ms:.1f}ms)")
        else:
            print(f"  ➜ Handoff: {transition} ({elapsed_ms:.1f}ms)")

    def log_tool(self, tool_name: str, status: str = "EXEC"):
        elapsed_ms = (time.perf_counter() - self.step_start_time) * 1000
        self.step_start_time = time.perf_counter()

        if console:
            console.print(f"  [magenta]⚙ Tool: {tool_name}[/magenta] ({elapsed_ms:.1f}ms)")
        else:
            print(f"  ⚙ Tool: {tool_name} ({elapsed_ms:.1f}ms)")

    def end_turn(self):
        total_ms = (time.perf_counter() - self.start_time) * 1000
        if console:
            console.print(f"  [dim]Turn completed in {total_ms:.1f}ms[/dim]\n")
        else:
            print(f"  Turn completed in {total_ms:.1f}ms\n")
        return total_ms
