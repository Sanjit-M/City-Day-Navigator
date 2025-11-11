import asyncio
import os
import json
import httpx
import re

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MCP_TOOLS_URL = "http://mcp-tools:3001"
MCP_API_KEY = os.getenv("MCP_API_KEY")
model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")


class PlanRequest(BaseModel):
    prompt: str


app = FastAPI(title="City Day Navigator - Orchestrator")


async def stream_plan(prompt: str):
    # Call Gemini (Classify)
    system_prompt = "You are an assistant that classifies user requests. Respond only with a JSON object. The user's date is 2025-11-10."
    user_prompt = (
        f"Classify this prompt: '{prompt}'. Your JSON must have these keys: "
        f"'intent' (one of 'plan_day', 'refine_plan', 'compare_options'), "
        f"'city' (string), 'date' (YYYY-MM-DD), 'country_code' (2-letter ISO), "
        f"'preferences' (list of strings)."
    )
    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'classify', 'status': 'pending'})}\n\n"
    try:
        response = await model.generate_content_async([system_prompt, user_prompt])
        text = getattr(response, "text", None) or ""
        classification = json.loads(text)
    except Exception as e:
        err = {'type': 'error', 'content': f'Classification failed: {str(e)}'}
        yield f"data: {json.dumps(err)}\n\n"
        return
    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'classify', 'status': 'complete', 'result': classification})}\n\n"

    # Handle intent
    if classification.get('intent') != 'plan_day':
        yield f"data: {json.dumps({'type': 'error', 'content': 'This demo only supports plan_day intent.'})}\n\n"
        return

    # Prepare HTTP client for MCP tools
    headers = {
        "X-API-KEY": MCP_API_KEY or "",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "CityDayNavigator-Orchestrator",
    }

    context: dict = {"classification": classification}
    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        # 1) Geocode first (to obtain lat/lon)
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
            yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'geocode', 'status': 'complete', 'result': geocode_data})}\n\n"
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
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-weather', 'fn': 'forecast', 'status': 'pending'})}\n\n"
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-air', 'fn': 'aqi', 'status': 'pending'})}\n\n"
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-calendar', 'fn': 'holidays', 'status': 'pending'})}\n\n"

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
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-weather', 'fn': 'forecast', 'status': 'complete', 'result': weather_data})}\n\n"
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-air', 'fn': 'aqi', 'status': 'complete', 'result': air_data})}\n\n"
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-calendar', 'fn': 'holidays', 'status': 'complete', 'result': holidays_data})}\n\n"

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
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues', 'status': 'pending'})}\n\n"
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
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'plan_venues', 'status': 'complete', 'result': venue_list})}\n\n"

        # Get Venue Details & ETAs
        itinerary_points: list[dict] = []
        for venue in venue_list:
            try:
                query = str(venue)
                nearby_req = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "query": query,
                    "radius_m": 3000,
                    "limit": 1,
                }
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'pending', 'query': query})}\n\n"
                r = await client.post(f"{MCP_TOOLS_URL}/geo/nearby", json=nearby_req)
                r.raise_for_status()
                nearby_data = r.json()
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-geo', 'fn': 'nearby', 'status': 'complete', 'result': nearby_data})}\n\n"
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
                    "profile": "foot",
                }
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'pending'})}\n\n"
                r_eta = await client.post(f"{MCP_TOOLS_URL}/route/eta", json=eta_req)
                r_eta.raise_for_status()
                eta_data = r_eta.json()
                context["eta"] = eta_data
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'complete', 'result': eta_data})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'mcp-route', 'fn': 'eta', 'status': 'error', 'error': str(e)})}\n\n"
        context["itinerary_points"] = itinerary_points

    # Call Gemini (Summarizer & Final Plan)
    final_system = (
        "You are a trip-planning assistant. Produce a complete, user-friendly Markdown itinerary. "
        "Follow these rules strictly: "
        "1) If rain probability > 60%, provide an indoor-heavy alternative section. "
        "2) If PM2.5 > 75 μg/m³, add 'mask recommended' to outdoor segments or suggest indoor swaps. "
        "3) If it's a public holiday, add a caution about closures or crowds. "
        "Include concise reasoning notes."
    )
    summary_context = {
        "classification": context.get("classification"),
        "weather": context.get("weather"),
        "air": context.get("air"),
        "holidays": context.get("holidays"),
        "eta": context.get("eta"),
        "itinerary_points": context.get("itinerary_points"),
    }
    summary_prompt = f"{final_system}\n\nContext:\n{json.dumps(summary_context)}\n\nGenerate the Markdown itinerary now."
    yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize', 'status': 'pending'})}\n\n"
    try:
        response_stream = await model.generate_content_async(summary_prompt, stream=True)
        async for chunk in response_stream:
            text = getattr(chunk, 'text', '') or ''
            if text:
                yield f"data: {json.dumps({'type': 'plan_chunk', 'content': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize', 'status': 'error', 'error': str(e)})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'tool_trace', 'service': 'gemini', 'fn': 'summarize', 'status': 'complete'})}\n\n"

    # Handle FX if requested in prompt
    lower_prompt = prompt.lower()
    fx_triggered = ("convert" in lower_prompt) or any(sym in prompt for sym in ["$", "€", "£", "¥", "₹"])
    if fx_triggered:
        # Naive parse for patterns like "Convert 200 USD to JPY"
        m = re.search(r'(\d+(?:\.\d+)?)\s*([A-Za-z]{3})\s*(?:to|in|->)\s*([A-Za-z]{3})', prompt, re.IGNORECASE)
        if m:
            amount = float(m.group(1))
            from_ccy = m.group(2).upper()
            to_ccy = m.group(3).upper()
            try:
                async with httpx.AsyncClient(timeout=10, headers=headers) as client_fx:
                    fx_body = {"amount": amount, "from": from_ccy, "to": to_ccy}
                    fx_resp = await client_fx.post(f"{MCP_TOOLS_URL}/fx/convert", json=fx_body)
                    fx_resp.raise_for_status()
                    fx_data = fx_resp.json()
                    out = {
                        "type": "plan_chunk",
                        "content": f"FX: {amount} {from_ccy} -> {to_ccy} = {fx_data.get('converted')} (rate {fx_data.get('rate')})",
                    }
                    yield f"data: {json.dumps(out)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'plan_chunk', 'content': f'FX conversion failed: {str(e)}'})}\n\n"

    # Signal completion
    yield "data: [DONE]\n\n"


@app.post("/plan-day")
async def plan_day(req: PlanRequest):
    return StreamingResponse(stream_plan(req.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3002)

