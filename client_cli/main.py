from __future__ import annotations

from typing import Optional
from pathlib import Path
import sys
import json
import os
import uuid

import typer
import rich
from rich.console import Console
import httpx


app = typer.Typer()
console = Console()
trace_console = Console(stderr=True)


@app.command()
def cli(
    prompt_str: Optional[str] = typer.Option(
        None, "--prompt", help="A full, free-form prompt to send to the orchestrator."
    ),
    city: Optional[str] = typer.Argument(None, help="The city for the itinerary (if not using --prompt)."),
    date: Optional[str] = typer.Argument(None, help="The date for the itinerary (YYYY-MM-DD, if not using --prompt)."),
    preferences: Optional[list[str]] = typer.Option(
        None, "--prefer", "-p", help="Preferences for the itinerary (e.g., 'museums', 'walkable')."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save Markdown to file."
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enter multi-turn interactive mode after the first response."
    ),
) -> None:
    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:3002")
    url = f"{orchestrator_url.rstrip('/')}/plan-day"
    session_id = str(uuid.uuid4())

    def run_once(one_prompt: str) -> str:
        markdown_output: str = ""
        with console.status("Planning your day..."):
            try:
                with httpx.stream(
                    "POST",
                    url,
                    json={"prompt": one_prompt, "session_id": session_id},
                    headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
                    timeout=60,
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith("data:"):
                            data = line[len("data:") :].strip()
                            if data == "[DONE]":
                                break
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                # Not JSON; ignore
                                continue

                            ptype = payload.get("type")
                            if ptype == "plan_chunk":
                                content = payload.get("content", "")
                                # Stream to console and collect for optional file output
                                console.print(content, end="")
                                markdown_output += content
                            elif ptype == "tool_trace":
                                service = payload.get("service")
                                fn = payload.get("fn")
                                status = payload.get("status")
                                dur = payload.get("duration_ms")
                                if dur is not None:
                                    try:
                                        dur_float = float(dur)
                                    except (TypeError, ValueError):
                                        dur_float = None
                                    if dur_float is not None:
                                        trace_console.print(f"[trace] {service}:{fn} -> {status} ({dur_float:.2f} ms)", style="dim")
                                    else:
                                        trace_console.print(f"[trace] {service}:{fn} -> {status}", style="dim")
                                else:
                                    trace_console.print(f"[trace] {service}:{fn} -> {status}", style="dim")
                            elif ptype == "error":
                                content = payload.get("content", "Unknown error")
                                trace_console.print(content, style="bold red")
                            # ignore other types
            except httpx.HTTPError as e:
                trace_console.print(f"Request failed: {e}", style="bold red")
        return markdown_output

    # Construct initial prompt from arguments (if any)
    initial_prompt: Optional[str] = None
    if prompt_str:
        initial_prompt = prompt_str
    elif city and date:
        initial_prompt = f"Plan a day in {city} on {date}."
        if preferences:
            initial_prompt += f" Preferences: {', '.join(preferences)}."

    # If no initial prompt and not interactive, prompt the user once for it
    if not initial_prompt and not interactive:
        # If no arguments supplied, fall back to a single interactive question
        try:
            initial_prompt = typer.prompt("Enter your request (e.g., 'Plan a day in Kyoto on 2025-12-12')")
        except (EOFError, KeyboardInterrupt):
            raise typer.Exit(code=1)
        # If user still provided nothing, exit
        if not (initial_prompt and initial_prompt.strip()):
            console.print("No input provided.", style="bold red")
            raise typer.Exit(code=1)

    # Run the initial request if present
    if initial_prompt:
        md = run_once(initial_prompt)
        if output_file:
            try:
                # Append results to output file (supports multiple turns)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("a", encoding="utf-8") as f:
                    if f.tell() > 0:
                        f.write("\n\n---\n\n")
                    f.write(md)
                console.print(f"\nSaved itinerary to {output_file}", style="green")
            except Exception as e:
                trace_console.print(f"Failed to write file: {e}", style="bold red")

    # Enter interactive loop if requested or if no initial prompt was given
    if interactive or (initial_prompt is None):
        while True:
            try:
                user_in = typer.prompt("Ask a follow-up or new request (type 'exit' to quit)")
            except (EOFError, KeyboardInterrupt):
                break
            if not user_in:
                continue
            lower = user_in.strip().lower()
            if lower in {"exit", "quit", "q"}:
                break
            md = run_once(user_in.strip())
            if output_file and md:
                try:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with output_file.open("a", encoding="utf-8") as f:
                        f.write("\n\n---\n\n")
                        f.write(md)
                    console.print(f"\nAppended output to {output_file}", style="green")
                except Exception as e:
                    trace_console.print(f"Failed to write file: {e}", style="bold red")


if __name__ == "__main__":
    # Some environments expect app() rather than app.run(); ensure compatibility.
    if not hasattr(app, "run"):
        app.run = app.__call__  # type: ignore[attr-defined]
    app.run()

