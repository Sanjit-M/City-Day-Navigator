import os
import re
import json
from typing import Tuple

import httpx


ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:3002").rstrip("/")
PLAN_ENDPOINT = f"{ORCHESTRATOR_URL}/plan-day"
DEFAULT_TIMEOUT = float(os.getenv("TEST_HTTP_TIMEOUT", "60"))


def _collect_sse_text(prompt: str) -> str:
    """
    Call the orchestrator /plan-day and collect only the concatenated plan_chunk text.
    """
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        with client.stream(
            "POST",
            PLAN_ENDPOINT,
            json={"prompt": prompt},
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            out: list[str] = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "plan_chunk":
                    out.append(payload.get("content", ""))
            return "".join(out)


def _contains_word(haystack: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", haystack, flags=re.IGNORECASE) is not None


def test_basic_plan():
    txt = _collect_sse_text("Plan 10:00–18:00 in Kyoto on 2025-12-12. Prefer temples and walkable.")
    # Weather note present
    assert _contains_word(txt, "Weather")
    # At least one ETA shown
    assert "ETA" in txt or "| ETA |" in txt or "Travel Leg" in txt
    # Has multiple distinct sections/headings (proxy for 4–6 stops)
    headings = len(re.findall(r"^###\s", txt, flags=re.MULTILINE))
    assert headings >= 4


def test_rain_fallback():
    # 1999-12-31 triggers heavy rain in weather tool
    txt = _collect_sse_text("Plan a day in Kyoto on 1999-12-31. Prefer museums and indoor.")
    assert re.search(r"alternative", txt, flags=re.IGNORECASE)
    assert re.search(r"indoor", txt, flags=re.IGNORECASE)


def test_air_guardrail():
    # 1999-12-30 triggers high PM2.5 in air tool
    txt = _collect_sse_text("Plan a walkable outdoor day in Delhi on 1999-12-30.")
    # Either explicitly mark mask recommended or suggest indoor swaps
    has_mask = re.search(r"mask recommended", txt, flags=re.IGNORECASE) is not None
    has_indoor_swap = re.search(r"indoor", txt, flags=re.IGNORECASE) is not None and re.search(r"swap|alternative", txt, flags=re.IGNORECASE) is not None
    assert has_mask or has_indoor_swap


def test_holiday_awareness():
    # Choose a well-known holiday date: US Independence Day
    txt = _collect_sse_text("Plan a day in New York on 2025-07-04. Prefer museums.")
    # Look for a holiday caution or closures/crowds note
    holiday_note = re.search(r"holiday", txt, flags=re.IGNORECASE) and re.search(r"caution|closure|crowd", txt, flags=re.IGNORECASE)
    assert holiday_note


def test_fx_only_short_circuit():
    txt = _collect_sse_text("Convert 200 USD to JPY")
    assert "Currency Conversion" in txt
    # Should not include an itinerary section
    assert "Itinerary" not in txt


def test_trace_visibility_table_present():
    txt = _collect_sse_text("Plan 10:00–18:00 in Kyoto on 2025-12-12. Prefer temples and walkable.")
    # Final tool trace summary table should be present
    assert "Tool trace summary" in txt
    # And should mention at least one known tool name
    assert "mcp-geo" in txt or "mcp-weather" in txt or "mcp-route" in txt


def test_determinism_repeat_prompt_same_order_and_eta():
    prompt = "Plan a museum-first day in Amsterdam on 2025-11-22. Bike preferred."
    a = _collect_sse_text(prompt)
    b = _collect_sse_text(prompt)
    # Ordering signal: Rijksmuseum before Van Gogh
    assert "Rijksmuseum" in a and "Van Gogh" in a
    assert "Rijksmuseum" in b and "Van Gogh" in b
    assert a.index("Rijksmuseum") < a.index("Van Gogh")
    assert b.index("Rijksmuseum") < b.index("Van Gogh")
    # Same ETA snippet appears in both (exact match acceptable due to cache)
    assert "3 minutes (0.3 km)" in a and "3 minutes (0.3 km)" in b


