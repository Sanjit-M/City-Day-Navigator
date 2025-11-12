import asyncio
import os
import json
import httpx
import re
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn


# In-memory session storage for interactive refinement
SESSIONS: Dict[str, Dict[str, Any]] = {}


CURRENCY_MAP = {
    # Symbols
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "฿": "THB",
    "A$": "AUD",
    "C$": "CAD",
    "CHF": "CHF",
    "CN¥": "CNY",
    "kr": "SEK", # Also NOK, DKK
    "NZ$": "NZD",
    "Mex$": "MXN",
    "S$": "SGD",
    "HK$": "HKD",
    "₺": "TRY",
    "₽": "RUB",
    "R$": "BRL",
    "R": "ZAR",
    # Names (lowercase)
    "dollar": "USD",
    "dollars": "USD",
    "usd": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "pound": "GBP",
    "pounds": "GBP",
    "gbp": "GBP",
    "yen": "JPY",
    "japanese yen": "JPY",
    "jpy": "JPY",
    "rupee": "INR",
    "rupees": "INR",
    "indian rupee": "INR",
    "inr": "INR",
    "won": "KRW",
    "korean won": "KRW",
    "krw": "KRW",
    "baht": "THB",
    "thai baht": "THB",
    "thb": "THB",
    "australian dollar": "AUD",
    "aud": "AUD",
    "canadian dollar": "CAD",
    "cad": "CAD",
    "swiss franc": "CHF",
    "chf": "CHF",
    "yuan": "CNY",
    "renminbi": "CNY",
    "cny": "CNY",
    "swedish krona": "SEK",
    "sek": "SEK",
    "new zealand dollar": "NZD",
    "nzd": "NZD",
    "mexican peso": "MXN",
    "mxn": "MXN",
    "singapore dollar": "SGD",
    "sgd": "SGD",
    "hong kong dollar": "HKD",
    "hkd": "HKD",
    "norwegian krone": "NOK",
    "nok": "NOK",
    "turkish lira": "TRY",
    "try": "TRY",
    "russian ruble": "RUB",
    "rub": "RUB",
    "brazilian real": "BRL",
    "brl": "BRL",
    "south african rand": "ZAR",
    "zar": "ZAR",
}

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MCP_TOOLS_URL = os.getenv("MCP_TOOLS_URL", "http://mcp-tools:3001")
MCP_API_KEY = os.getenv("MCP_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")
CLASSIFICATION_DEFAULT_DATE = os.getenv("CLASSIFICATION_DEFAULT_DATE", "2025-11-10")
ROUTE_PROFILE_DEFAULT = os.getenv("ROUTE_PROFILE_DEFAULT", "foot")
NEARBY_RADIUS_M = int(os.getenv("NEARBY_RADIUS_M", "3000"))
NEARBY_LIMIT = int(os.getenv("NEARBY_LIMIT", "1"))
try:
    HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "15"))
except ValueError:
    HTTP_TIMEOUT_SEC = 15.0
model = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config={"response_mime_type": "application/json"},
)
# New model for text generation
text_model = genai.GenerativeModel(GEMINI_MODEL)


class PlanRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


app = FastAPI(title="City Day Navigator - Orchestrator")


async def stream_plan(prompt: str, session_id: Optional[str]):
    def normalize_currency(s: str) -> Optional[str]:
        """Attempt to map a currency phrase to a 3-letter ISO code."""
        if not s:
            return None
        s_norm = s.strip().lower()
        code = CURRENCY_MAP.get(s_norm)
        if code and len(code) == 3:
            return code
        # Try token-wise lookup (e.g., 'japanese yen' -> 'yen' -> 'JPY')
        for token in reversed(re.split(r"[\s\-_]+", s_norm)):
            if not token:
                continue
            code = CURRENCY_MAP.get(token)
            if code and len(code) == 3:
                return code
        # If already a 3-letter code, normalize upper
        if len(s_norm) == 3 and s_norm.isalpha():
            return s_norm.upper()
        return None
    tool_traces = []  # collect completed tool calls with durations
    # Call Gemini (Classify)
    system_prompt = f"You are an assistant that classifies user requests. Respond only with a JSON object. The user's date is {CLASSIFICATION_DEFAULT_DATE}."
    user_prompt = (
        f"Classify this prompt: '{prompt}'. Your JSON must have these keys: "
        f"'intent' (one of 'plan_day', 'refine_plan', 'compare_options'), "
        f"'city' (string), 'date' (YYYY-MM-DD), 'country_code' (2-letter ISO), "
        f"'preferences' (list of strings)."
    )
    t0 = time.monotonic()
    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'classify', 'status': 'pending'})}\n\n"
    try:
        response = await model.generate_content_async([system_prompt, user_prompt])
        text = getattr(response, "text", None) or ""
        try:
            classification = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise
            classification = json.loads(m.group(0))
    except Exception as e:
        err = {'type': 'error', 'content': f'Classification failed: {str(e)}'}
        yield f"data: {json.dumps(err)}\n\n"
        return
    duration_ms = (time.monotonic() - t0) * 1000
    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'classify', 'status': 'complete', 'duration_ms': f'{duration_ms:.2f}', 'result': classification})}\n\n"
    tool_traces.append({'service': 'gemini', 'fn': 'classify', 'duration_ms': duration_ms})

    # Prepare HTTP client for MCP tools
    headers = {
        "X-API-KEY": MCP_API_KEY or "",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "CityDayNavigator-Orchestrator",
    }
    context: dict = {"classification": classification}

    # Handle standalone FX query BEFORE 'plan_day' check
    lower_prompt = prompt.lower()
    fx_keywords = ["convert", "exchange", "fx", "rate"]
    currency_symbols_or_codes = [
        "$", "€", "£", "¥", "₹", "₩", "฿", "A$", "C$", "CHF", "CN¥", "kr", "NZ$",
        "Mex$", "S$", "HK$", "₺", "₽", "R$", "R",
        "usd", "eur", "gbp", "jpy", "inr", "krw", "thb", "aud", "cad", "chf",
        "cny", "sek", "nzd", "mxn", "sgd", "hkd", "nok", "try", "rub", "brl", "zar",
    ]
    fx_triggered = any(kw in lower_prompt for kw in fx_keywords) or any(c in lower_prompt for c in currency_symbols_or_codes)

    if fx_triggered and classification.get('intent') != 'plan_day':
        currency_pattern = r"([A-Za-z\s]{3,}|[\$€£¥₹₩฿₽₺]|A\$|C\$|NZ\$|Mex\$|S\$|HK\$|R\$|R|kr|CHF|CN¥)"
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*" + currency_pattern + r"\s*(?:to|in|->|=)\s*" + currency_pattern,
            prompt,
            re.IGNORECASE,
        )
        if m:
            try:
                async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC, headers=headers) as client:
                    amount_str, from_str, to_str = m.group(1), m.group(2).strip(), m.group(3).strip()
                    amount = float(amount_str)
                    from_ccy = normalize_currency(from_str)
                    to_ccy = normalize_currency(to_str)

                    if len(from_ccy) == 3 and len(to_ccy) == 3:
                        t_fx = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-fx', 'fn': 'convert', 'status': 'pending'})}\n\n"
                        fx_body = {"amount": amount, "from": from_ccy, "to": to_ccy}
                        fx_resp = await client.post(f"{MCP_TOOLS_URL}/fx/convert", json=fx_body)
                        fx_resp.raise_for_status()
                        fx_data = fx_resp.json()
                        fx_dur_ms = (time.monotonic()-t_fx)*1000
                        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-fx', 'fn': 'convert', 'status': 'complete', 'duration_ms': f'{fx_dur_ms:.2f}', 'result': fx_data})}\n\n"
                        tool_traces.append({'service': 'mcp-fx', 'fn': 'convert', 'duration_ms': fx_dur_ms})

                        rate = fx_data.get('rate', 'N/A')
                        converted = fx_data.get('converted', 'N/A')
                        summary = (
                            f"## Currency Conversion\n\n"
                            f"| From | To | Rate | Result |\n"
                            f"| :--- | :--- | :--- | :--- |\n"
                            f"| {amount} {from_ccy} | {to_ccy} | {rate} | **{converted} {to_ccy}** |\n"
                        )
                        yield f"data: {json.dumps({'type': 'plan_chunk', 'content': summary})}\n\n"
                        # Emit trace summary
                        if tool_traces:
                            summary_md = "### Tool trace summary\n\n| Service | Function | Duration (ms) |\n| :--- | :--- | ---: |\n"
                            for tr in tool_traces:
                                summary_md += f"| {tr['service']} | {tr['fn']} | {tr['duration_ms']:.2f} |\n"
                            yield f"data: {json.dumps({'type': 'plan_chunk', 'content': summary_md})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

            except Exception as e:
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-fx', 'fn': 'convert', 'status': 'error', 'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
                return

    # Handle non-plan intents with session support
    intent = classification.get('intent')
    if intent in ('refine_plan', 'compare_options'):
        session_ctx = None
        if session_id:
            session_ctx = SESSIONS.get(session_id)
        if not session_ctx:
            # Fallback: if no prior context, downgrade to planning a fresh day
            intent = 'plan_day'
        else:
            # We have prior context; branch per intent
            if intent == 'compare_options':
                # Produce two alternatives using existing context; no new tool calls required
                summary_context = {
                    "classification": session_ctx.get("classification"),
                    "weather": session_ctx.get("weather"),
                    "air": session_ctx.get("air"),
                    "holidays": session_ctx.get("holidays"),
                    "eta": session_ctx.get("eta"),
                    "itinerary_points": session_ctx.get("itinerary_points"),
                    "fx": session_ctx.get("fx"),
                }
                compare_system = (
                    "You are a trip-planning assistant. Produce TWO alternative Markdown itineraries labeled "
                    "'Option A' and 'Option B' for the same day and city using the provided context. "
                    "Keep all formatting and guardrails (ETAs, indoor-alternative for rain > 60%, AQI caveats, holiday notes). "
                    "Apply the user's comparison/refinement instruction to differentiate the options."
                )
                summary_prompt = (
                    f"{compare_system}\n\n"
                    f"User instruction: {prompt}\n\n"
                    f"Context:\n{json.dumps(summary_context)}\n\n"
                    "Generate the two Markdown options now."
                )
                t_sum = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize_compare', 'status': 'pending'})}\n\n"
                try:
                    response_stream = await text_model.generate_content_async(summary_prompt, stream=True)
                    async for chunk in response_stream:
                        text = getattr(chunk, 'text', '') or ''
                        if text:
                            yield f"data: {json.dumps({'type': 'plan_chunk', 'content': text})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize_compare', 'status': 'error', 'error': str(e)})}\n\n"
                else:
                    sc_ms = (time.monotonic()-t_sum)*1000
                    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize_compare', 'status': 'complete', 'duration_ms': f'{sc_ms:.2f}'})}\n\n"
                    tool_traces.append({'service': 'gemini', 'fn': 'summarize_compare', 'duration_ms': sc_ms})
                if tool_traces:
                    summary_md = "### Tool trace summary\n\n| Service | Function | Duration (ms) |\n| :--- | :--- | ---: |\n"
                    for tr in tool_traces:
                        summary_md += f"| {tr['service']} | {tr['fn']} | {tr['duration_ms']:.2f} |\n"
                    yield f"data: {json.dumps({'type': 'plan_chunk', 'content': summary_md})}\n\n"
                yield "data: [DONE]\n\n"
                return
            elif intent == 'refine_plan':
                # Re-plan venues with refinement, then proceed similarly to plan_day flow using prior geocode/weather/air/holidays
                lat = session_ctx.get("geocode", {}).get("lat")
                lon = session_ctx.get("geocode", {}).get("lon")
                date = session_ctx.get("classification", {}).get("date")
                country_code = session_ctx.get("classification", {}).get("country_code")
                prefs = classification.get("preferences") or session_ctx.get("classification", {}).get("preferences") or []
                if lat is None or lon is None:
                    # If for some reason geocode missing, drop to fresh plan
                    intent = 'plan_day'
                else:
                    # Build refinement-aware planner prompt including previous venues
                    previous_points = session_ctx.get("itinerary_points") or []
                    previous_names = [p.get("name") for p in previous_points if p.get("name")]
                    planning_context = {
                        "weather": session_ctx.get("weather"),
                        "air": session_ctx.get("air"),
                        "holidays": session_ctx.get("holidays"),
                        "preferences": prefs,
                        "city": session_ctx.get("classification", {}).get("city"),
                        "date": date,
                        "country_code": country_code,
                        "current_venues": previous_names,
                        "refinement_instruction": prompt,
                    }
                    planner_prompt = (
                        "Refine the existing itinerary's venue list based on the refinement instruction while keeping "
                        "the day coherent. Respond ONLY with a JSON array of 4-6 venue names/types. "
                        "If the instruction mentions a relative position (e.g., 'second venue'), interpret it relative "
                        "to current_venues and adjust accordingly.\n\n"
                        f"Context:\n{json.dumps(planning_context)}"
                    )
                    t_plan = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues_refine', 'status': 'pending'})}\n\n"
                    try:
                        venues_response = await model.generate_content_async(planner_prompt)
                        venues_text = getattr(venues_response, "text", None) or ""
                        venue_list = json.loads(venues_text)
                        if not isinstance(venue_list, list):
                            raise ValueError("Planner did not return a list")
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues_refine', 'status': 'error', 'error': str(e)})}\n\n"
                        yield f"data: {json.dumps({'type': 'error', 'content': 'Refinement planning failed.'})}\n\n"
                        return
                    pv_r_ms = (time.monotonic()-t_plan)*1000
                    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues_refine', 'status': 'complete', 'duration_ms': f'{pv_r_ms:.2f}', 'result': venue_list})}\n\n"
                    tool_traces.append({'service': 'gemini', 'fn': 'plan_venues_refine', 'duration_ms': pv_r_ms})

                    # Nearby + ETA based on refined venues
                    headers = {
                        "X-API-KEY": MCP_API_KEY or "",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "CityDayNavigator-Orchestrator",
                    }
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC, headers=headers) as client:
                        itinerary_points: list[dict] = []
                        for venue in venue_list:
                            try:
                                query = str(venue)
                                nearby_req = {
                                    "lat": float(lat),
                                    "lon": float(lon),
                                    "query": query,
                                    "radius_m": NEARBY_RADIUS_M,
                                    "limit": NEARBY_LIMIT,
                                }
                                t_nb = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'pending', 'query': query})}\n\n"
                                r = await client.post(f"{MCP_TOOLS_URL}/geo/nearby", json=nearby_req)
                                r.raise_for_status()
                                nearby_data = r.json()
                                nb_ms = (time.monotonic()-t_nb)*1000
                                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'complete', 'duration_ms': f'{nb_ms:.2f}', 'result': nearby_data})}\n\n"
                                tool_traces.append({'service': 'mcp-geo', 'fn': 'nearby', 'duration_ms': nb_ms})
                                items = nearby_data.get("results") or []
                                if not items:
                                    continue
                                top = items[0]
                                itinerary_points.append({"name": query, "lat": top.get("lat"), "lon": top.get("lon")})
                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'error', 'error': str(e), 'query': str(venue)})}\n\n"
                                continue

                        eta_data = None
                        if len(itinerary_points) >= 2:
                            try:
                                eta_req = {
                                    "points": [{"lat": p["lat"], "lon": p["lon"]} for p in itinerary_points],
                                    "profile": ROUTE_PROFILE_DEFAULT,
                                }
                                t_eta = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'pending'})}\n\n"
                                r_eta = await client.post(f"{MCP_TOOLS_URL}/route/eta", json=eta_req)
                                r_eta.raise_for_status()
                                eta_data = r_eta.json()
                                eta_ms = (time.monotonic()-t_eta)*1000
                                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'complete', 'duration_ms': f'{eta_ms:.2f}', 'result': eta_data})}\n\n"
                                tool_traces.append({'service': 'mcp-route', 'fn': 'eta', 'duration_ms': eta_ms})
                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'error', 'error': str(e)})}\n\n"

                    # Update session context and summarize refined plan
                    new_context = {
                        "classification": session_ctx.get("classification"),
                        "weather": session_ctx.get("weather"),
                        "air": session_ctx.get("air"),
                        "holidays": session_ctx.get("holidays"),
                        "eta": eta_data,
                        "itinerary_points": itinerary_points,
                        "fx": session_ctx.get("fx"),
                        "geocode": session_ctx.get("geocode"),
                    }
                    if session_id:
                        SESSIONS[session_id] = new_context

                    final_system = (
                        "You are a trip-planning assistant. Produce a complete, user-friendly Markdown itinerary. "
                        "Follow these rules strictly: "
                        "1) Insert travel legs with ETAs between itinerary points. "
                        "2) If rain probability > 60%, provide an indoor-heavy alternative section. "
                        "3) If PM2.5 > 75 μg/m³, add 'mask recommended' to outdoor segments or suggest indoor swaps. "
                        "4) If it's a public holiday, add a caution about closures or crowds. "
                        "5) If the context includes 'fx' data, add a formatted 'Currency Conversion' section. "
                        "Include concise reasoning notes. State applied refinement explicitly."
                    )
                    summary_context = new_context
                    summary_prompt = f"{final_system}\n\nApplied refinement: {prompt}\n\nContext:\n{json.dumps(summary_context)}\n\nGenerate the Markdown itinerary now."
                    t_sum = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize_refined', 'status': 'pending'})}\n\n"
                    try:
                        response_stream = await text_model.generate_content_async(summary_prompt, stream=True)
                        async for chunk in response_stream:
                            text = getattr(chunk, 'text', '') or ''
                            if text:
                                yield f"data: {json.dumps({'type': 'plan_chunk', 'content': text})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize_refined', 'status': 'error', 'error': str(e)})}\n\n"
                    else:
                        sr_ms = (time.monotonic()-t_sum)*1000
                        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize_refined', 'status': 'complete', 'duration_ms': f'{sr_ms:.2f}'})}\n\n"
                        tool_traces.append({'service': 'gemini', 'fn': 'summarize_refined', 'duration_ms': sr_ms})
                    if tool_traces:
                        summary_md = "### Tool trace summary\n\n| Service | Function | Duration (ms) |\n| :--- | :--- | ---: |\n"
                        for tr in tool_traces:
                            summary_md += f"| {tr['service']} | {tr['fn']} | {tr['duration_ms']:.2f} |\n"
                        yield f"data: {json.dumps({'type': 'plan_chunk', 'content': summary_md})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC, headers=headers) as client:
        # 1) Geocode first (to obtain lat/lon)
        t1 = time.monotonic()
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'geocode', 'status': 'pending'})}\n\n"
        try:
            geo_req = {
                "city": classification.get("city"),
                "country_hint": classification.get("country_code"),
            }
            geo_resp = await client.post(f"{MCP_TOOLS_URL}/geo/geocode", json=geo_req)
            geo_resp.raise_for_status()
            geocode_data = geo_resp.json()
            context["geocode"] = geocode_data
            duration_ms = (time.monotonic() - t1) * 1000
            yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'geocode', 'status': 'complete', 'duration_ms': f'{duration_ms:.2f}', 'result': geocode_data})}\n\n"
            tool_traces.append({'service': 'mcp-geo', 'fn': 'geocode', 'duration_ms': duration_ms})
        except Exception as e:
            yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'geocode', 'status': 'error', 'error': str(e)})}\n\n"
            return

        lat = context["geocode"].get("lat")
        lon = context["geocode"].get("lon")
        date = classification.get("date")
        country_code = classification.get("country_code")
        try:
            year = int((date or "1970-01-01").split("-")[0])
        except Exception:
            year = 1970

        # 2) Weather, Air, Holidays in parallel
        # Emit pending traces
        t_weather = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-weather', 'fn': 'forecast', 'status': 'pending'})}\n\n"
        t_air = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-air', 'fn': 'aqi', 'status': 'pending'})}\n\n"
        t_hol = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-calendar', 'fn': 'holidays', 'status': 'pending'})}\n\n"

        async def call_weather():
            try:
                w_req = {"lat": float(lat), "lon": float(lon), "date": date}
                r = await client.post(f"{MCP_TOOLS_URL}/weather/forecast", json=w_req)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                return {"error": str(e)}

        async def call_air():
            try:
                a_req = {"lat": float(lat), "lon": float(lon), "date": date}
                r = await client.post(f"{MCP_TOOLS_URL}/air/aqi", json=a_req)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                return {"error": str(e)}

        async def call_holidays():
            try:
                h_req = {"country_code": country_code, "year": year}
                r = await client.post(f"{MCP_TOOLS_URL}/calendar/holidays", json=h_req)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                return {"error": str(e)}

        weather_data, air_data, holidays_data = await asyncio.gather(
            call_weather(), call_air(), call_holidays()
        )
        context["weather"] = weather_data
        context["air"] = air_data
        context["holidays"] = holidays_data

        # Emit complete traces
        w_ms = (time.monotonic()-t_weather)*1000
        a_ms = (time.monotonic()-t_air)*1000
        h_ms = (time.monotonic()-t_hol)*1000
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-weather', 'fn': 'forecast', 'status': 'complete', 'duration_ms': f'{w_ms:.2f}', 'result': weather_data})}\n\n"
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-air', 'fn': 'aqi', 'status': 'complete', 'duration_ms': f'{a_ms:.2f}', 'result': air_data})}\n\n"
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-calendar', 'fn': 'holidays', 'status': 'complete', 'duration_ms': f'{h_ms:.2f}', 'result': holidays_data})}\n\n"
        tool_traces.append({'service': 'mcp-weather', 'fn': 'forecast', 'duration_ms': w_ms})
        tool_traces.append({'service': 'mcp-air', 'fn': 'aqi', 'duration_ms': a_ms})
        tool_traces.append({'service': 'mcp-calendar', 'fn': 'holidays', 'duration_ms': h_ms})

        # Call Gemini (Planner) to propose venues
        prefs = classification.get("preferences") or []
        planning_context = {
            "weather": context.get("weather"),
            "air": context.get("air"),
            "holidays": context.get("holidays"),
            "preferences": prefs,
            "city": classification.get("city"),
            "date": classification.get("date"),
            "country_code": classification.get("country_code"),
        }
        planner_prompt = (
            "Given this context for planning a day itinerary, propose a JSON array of 4-6 venue names or venue types "
            "(e.g., ['Kinkaku-ji Temple', 'Nishiki Market', 'Coffee shop']) that match user preferences and constraints. "
            "Prefer indoor options if precipitation probability > 60%. Respond ONLY with a JSON array of strings.\n\n"
            f"Context:\n{json.dumps(planning_context)}"
        )
        t_plan = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues', 'status': 'pending'})}\n\n"
        try:
            venues_response = await model.generate_content_async(planner_prompt)
            venues_text = getattr(venues_response, "text", None) or ""
            venue_list = json.loads(venues_text)
            if not isinstance(venue_list, list):
                raise ValueError("Planner did not return a list")
        except Exception as e:
            yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues', 'status': 'error', 'error': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'content': 'Planning failed.'})}\n\n"
            return
        pv_ms = (time.monotonic()-t_plan)*1000
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues', 'status': 'complete', 'duration_ms': f'{pv_ms:.2f}', 'result': venue_list})}\n\n"
        tool_traces.append({'service': 'gemini', 'fn': 'plan_venues', 'duration_ms': pv_ms})

        # Get Venue Details & ETAs
        itinerary_points: list[dict] = []
        for venue in venue_list:
            try:
                query = str(venue)
                nearby_req = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "query": query,
                    "radius_m": NEARBY_RADIUS_M,
                    "limit": NEARBY_LIMIT,
                }
                t_nb = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'pending', 'query': query})}\n\n"
                r = await client.post(f"{MCP_TOOLS_URL}/geo/nearby", json=nearby_req)
                r.raise_for_status()
                nearby_data = r.json()
                nb_ms = (time.monotonic()-t_nb)*1000
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'complete', 'duration_ms': f'{nb_ms:.2f}', 'result': nearby_data})}\n\n"
                tool_traces.append({'service': 'mcp-geo', 'fn': 'nearby', 'duration_ms': nb_ms})
                items = nearby_data.get("results") or []
                if not items:
                    continue
                top = items[0]
                itinerary_points.append({"name": query, "lat": top.get("lat"), "lon": top.get("lon")})
            except Exception as e:
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'error', 'error': str(e), 'query': str(venue)})}\n\n"
                continue

        # If we have at least 2 points, compute ETA across them
        if len(itinerary_points) >= 2:
            try:
                eta_req = {
                    "points": [{"lat": p["lat"], "lon": p["lon"]} for p in itinerary_points],
                    "profile": ROUTE_PROFILE_DEFAULT,
                }
                t_eta = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'pending'})}\n\n"
                r_eta = await client.post(f"{MCP_TOOLS_URL}/route/eta", json=eta_req)
                r_eta.raise_for_status()
                eta_data = r_eta.json()
                context["eta"] = eta_data
                eta_ms = (time.monotonic()-t_eta)*1000
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'complete', 'duration_ms': f'{eta_ms:.2f}', 'result': eta_data})}\n\n"
                tool_traces.append({'service': 'mcp-route', 'fn': 'eta', 'duration_ms': eta_ms})
            except Exception as e:
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'error', 'error': str(e)})}\n\n"
        context["itinerary_points"] = itinerary_points

        # Handle FX if requested in prompt
        lower_prompt = prompt.lower()
        fx_keywords = ["convert", "exchange", "fx", "rate"]
        currency_symbols_or_codes = [
            "$", "€", "£", "¥", "₹", "₩", "฿", "A$", "C$", "CHF", "CN¥", "kr", "NZ$",
            "Mex$", "S$", "HK$", "₺", "₽", "R$", "R",
            "usd", "eur", "gbp", "jpy", "inr", "krw", "thb", "aud", "cad", "chf",
            "cny", "sek", "nzd", "mxn", "sgd", "hkd", "nok", "try", "rub", "brl", "zar",
        ]
        fx_triggered = any(kw in lower_prompt for kw in fx_keywords) or any(c in lower_prompt for c in currency_symbols_or_codes)
        if fx_triggered:
            # More robust currency pattern
            currency_pattern = r"([A-Za-z\s]{3,}|[\$€£¥₹₩฿₽₺]|A\$|C\$|NZ\$|Mex\$|S\$|HK\$|R\$|R|kr|CHF|CN¥)"
            m = re.search(
                r"(\d+(?:\.\d+)?)\s*" + currency_pattern + r"\s*(?:to|in|->|=)\s*" + currency_pattern,
                prompt,
                re.IGNORECASE,
            )
            if m:
                try:
                    amount_str, from_str, to_str = m.group(1), m.group(2).strip(), m.group(3).strip()
                    amount = float(amount_str)
                    from_ccy = normalize_currency(from_str)
                    to_ccy = normalize_currency(to_str)

                    if len(from_ccy) == 3 and len(to_ccy) == 3:
                        t_fx = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-fx', 'fn': 'convert', 'status': 'pending'})}\n\n"
                        fx_body = {"amount": amount, "from": from_ccy, "to": to_ccy}
                        fx_resp = await client.post(f"{MCP_TOOLS_URL}/fx/convert", json=fx_body)
                        fx_resp.raise_for_status()
                        fx_data = fx_resp.json()
                        context["fx"] = fx_data
                        fx_ms = (time.monotonic()-t_fx)*1000
                        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-fx', 'fn': 'convert', 'status': 'complete', 'duration_ms': f'{fx_ms:.2f}', 'result': fx_data})}\n\n"
                        tool_traces.append({'service': 'mcp-fx', 'fn': 'convert', 'duration_ms': fx_ms})
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-fx', 'fn': 'convert', 'status': 'error', 'error': str(e)})}\n\n"


    # Store/refresh session context for future refinements
    if session_id:
        SESSIONS[session_id] = {
            "classification": context.get("classification"),
            "weather": context.get("weather"),
            "air": context.get("air"),
            "holidays": context.get("holidays"),
            "eta": context.get("eta"),
            "itinerary_points": context.get("itinerary_points"),
            "fx": context.get("fx"),
            "geocode": context.get("geocode"),
        }

    # Call Gemini (Summarizer & Final Plan)
    final_system = (
        "You are a trip-planning assistant. Produce a complete, user-friendly Markdown itinerary. "
        "Follow these rules strictly: "
        "1) Insert travel legs with ETAs between itinerary points. "
        "2) If rain probability > 60%, provide an indoor-heavy alternative section. "
        "3) If PM2.5 > 75 μg/m³, add 'mask recommended' to outdoor segments or suggest indoor swaps. "
        "4) If it's a public holiday, add a caution about closures or crowds. "
        "5) If the context includes 'fx' data, add a formatted 'Currency Conversion' section. "
        "Include concise reasoning notes."
    )
    summary_context = {
        "classification": context.get("classification"),
        "weather": context.get("weather"),
        "air": context.get("air"),
        "holidays": context.get("holidays"),
        "eta": context.get("eta"),
        "itinerary_points": context.get("itinerary_points"),
        "fx": context.get("fx"),
    }
    summary_prompt = f"{final_system}\n\nContext:\n{json.dumps(summary_context)}\n\nGenerate the Markdown itinerary now."
    t_sum = time.monotonic(); yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize', 'status': 'pending'})}\n\n"
    try:
        response_stream = await text_model.generate_content_async(summary_prompt, stream=True)
        async for chunk in response_stream:
            text = getattr(chunk, 'text', '') or ''
            if text:
                yield f"data: {json.dumps({'type': 'plan_chunk', 'content': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize', 'status': 'error', 'error': str(e)})}\n\n"
    else:
        sum_ms = (time.monotonic()-t_sum)*1000
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize', 'status': 'complete', 'duration_ms': f'{sum_ms:.2f}'})}\n\n"
        tool_traces.append({'service': 'gemini', 'fn': 'summarize', 'duration_ms': sum_ms})

    # Signal completion
    if tool_traces:
        summary_md = "### Tool trace summary\n\n| Service | Function | Duration (ms) |\n| :--- | :--- | ---: |\n"
        for tr in tool_traces:
            summary_md += f"| {tr['service']} | {tr['fn']} | {tr['duration_ms']:.2f} |\n"
        yield f"data: {json.dumps({'type': 'plan_chunk', 'content': summary_md})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/plan-day")
async def plan_day(req: PlanRequest):
    return StreamingResponse(stream_plan(req.prompt, req.session_id), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3002"))
    uvicorn.run(app, host="0.0.0.0", port=port)

