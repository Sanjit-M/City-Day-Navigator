from __future__ import annotations

from typing import Optional
from pathlib import Path
import sys
import json
import os

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
) -> None:
    markdown_output = ""
    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:3002")
    url = f"{orchestrator_url.rstrip('/')}/plan-day"

    # Construct prompt from arguments
    final_prompt = ""
    if prompt_str:
        final_prompt = prompt_str
    elif city and date:
        final_prompt = f"Plan a day in {city} on {date}."
        if preferences:
            final_prompt += f" Preferences: {', '.join(preferences)}."
    else:
        console.print(
            "Error: You must provide either a full prompt with --prompt, or a city and date.",
            style="bold red",
        )
        raise typer.Exit(code=1)

    with console.status("Planning your day..."):
        try:
            with httpx.stream(
                "POST",
                url,
                json={"prompt": final_prompt},
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
                            trace_console.print(f"[trace] {service}:{fn} -> {status}", style="dim")
                        elif ptype == "error":
                            content = payload.get("content", "Unknown error")
                            trace_console.print(content, style="bold red")
                        # ignore other types
        except httpx.HTTPError as e:
            trace_console.print(f"Request failed: {e}", style="bold red")

    if output_file:
        try:
            output_file.write_text(markdown_output, encoding="utf-8")
            console.print(f"\nSaved itinerary to {output_file}", style="green")
        except Exception as e:
            trace_console.print(f"Failed to write file: {e}", style="bold red")


if __name__ == "__main__":
    # Some environments expect app() rather than app.run(); ensure compatibility.
    if not hasattr(app, "run"):
        app.run = app.__call__  # type: ignore[attr-defined]
    app.run()

