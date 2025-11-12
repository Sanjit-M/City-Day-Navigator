## City Day Navigator

Plan a compact, weather-aware day itinerary for any city. The system:
- Classifies the request with Gemini
- Calls MCP tools (geo, weather, air, calendar, route, fx)
- Streams a concise Markdown itinerary with travel legs, guardrails (rain/AQI), and optional currency conversion


## 1) Setup, run, and example prompts

### Prerequisites
- Docker and docker-compose
- Python 3.11+ for the local CLI (optional)
- API keys (place these in a `.env` file at the repo root):
  - MCP_API_KEY=dev-key (or your own)
  - GEMINI_API_KEY=...
  - EXCHANGERATE_API_KEY=...
  - OPENAQ_API_KEY=...

Optional overrides (default values shown):
- GEMINI_MODEL=gemini-2.5-flash-preview-09-2025
- CLASSIFICATION_DEFAULT_DATE=2025-11-10
- HTTP_TIMEOUT_SEC=15
- NEARBY_RADIUS_M=3000
- NEARBY_LIMIT=1
- ROUTE_PROFILE_DEFAULT=foot


### Make targets (optional)
```makefile
start:
	docker-compose up --build -d

restart:
	docker-compose down && docker-compose up --build -d

stop:
	docker-compose down

ps:
	docker-compose ps

logs:
	docker-compose logs -f mcp_tools orchestrator
```


Services:
- MCP Tools API on http://localhost:3001 (requires header `X-API-KEY`)
- Orchestrator API on http://localhost:3002


### CLI client
Create a virtualenv and install the CLI dependencies with uv:
```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r client_cli/requirements.txt
```

Run the CLI with structured args:
```bash
source .venv/bin/activate
python3 client_cli/main.py "Kyoto" "2025-12-12" --prefer "temples" --prefer "walkable"
```

Run the CLI with a free-form prompt:
```bash
source .venv/bin/activate
python3 client_cli/main.py --prompt "Plan 10:00–18:00 in Kyoto on 2025-12-12. Prefer temples and walkable"
```

Include a currency conversion in the prompt (orchestrator will call MCP FX):
```bash
source .venv/bin/activate
python3 client_cli/main.py --prompt "Plan a day in Tokyo on 2025-11-20. Also convert 200 USD to JPY"
```

You will see streaming tool traces in stderr and Markdown in stdout.


### Quick health checks
```bash
curl -s -H "X-API-KEY: ${MCP_API_KEY:-dev-key}" http://localhost:3001/ | jq
curl -s http://localhost:3002/plan-day -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Plan a museum-first day in Amsterdam on 2025-11-22. Bike preferred."}'
```


## 2) API endpoints used and request examples

### External APIs (free/no-key or public tiers)
- Open-Meteo (weather forecast):
  - Base: `https://api.open-meteo.com/v1/forecast`
  - Example:
    ```bash
    curl -G "https://api.open-meteo.com/v1/forecast" \
      --data-urlencode "latitude=35.0116" \
      --data-urlencode "longitude=135.7681" \
      --data-urlencode "hourly=temperature_2m,precipitation_probability,windspeed_10m" \
      --data-urlencode "timezone=UTC"
    ```
- OpenAQ v3 (air quality observations):
  - Base: `https://api.openaq.org/v3`
  - Example (latest near coords, repeated parameters):
    ```bash
    curl -G "https://api.openaq.org/v3/latest" \
      -H "X-API-Key: $OPENAQ_API_KEY" \
      --data-urlencode "coordinates=40.7128,-74.0060" \
      --data-urlencode "radius=25000" \
      --data-urlencode "limit=50" \
      --data-urlencode "parameter=pm25" \
      --data-urlencode "parameter=pm10" \
      --data-urlencode "parameter=no2" \
      --data-urlencode "parameter=o3"
    ```
- Nominatim / OpenStreetMap (geocoding + places):
  - `https://nominatim.openstreetmap.org/search`
  - `https://nominatim.openstreetmap.org/reverse`
- OSRM demo (routing and ETAs):
  - `https://router.project-osrm.org/route/v1/{profile}/{lon},{lat};{lon},{lat}`
- Nager.Date (public holidays):
  - `https://date.nager.at/api/v3/PublicHolidays/{year}/{countryCode}`
- exchangerate.host (FX rates):
  - Latest: `https://api.exchangerate.host/latest`
  - Convert: `https://api.exchangerate.host/convert?from=USD&to=JPY&amount=200`
- Open‑Meteo Air Quality (fallback if OpenAQ has no nearby observations):
  - `https://air-quality-api.open-meteo.com/v1/air-quality`


### MCP Tools (local) and examples
All MCP endpoints require `X-API-KEY` header.

- Geocoding
  - POST `http://localhost:3001/geo/geocode`
  - Body: `{ "city": "Kyoto", "country_hint": "JP" }`
- Nearby places (Nominatim search)
  - POST `http://localhost:3001/geo/nearby`
  - Body: `{ "lat": 35.0116, "lon": 135.7681, "query": "coffee", "radius_m": 3000, "limit": 3 }`
- Weather forecast (Open-Meteo)
  - POST `http://localhost:3001/weather/forecast`
  - Body: `{ "lat": 35.0116, "lon": 135.7681, "date": "2025-12-12" }`
- Air quality (OpenAQ v3 with Open‑Meteo fallback)
  - POST `http://localhost:3001/air/aqi`
  - Body: `{ "lat": 40.7128, "lon": -74.0060 }`
- Routing/ETA (OSRM)
  - POST `http://localhost:3001/route/eta`
  - Body: `{ "points": [{"lat": 35.0,"lon":135.7},{"lat":35.02,"lon":135.76}], "profile": "foot" }`
- Public holidays (Nager.Date)
  - POST `http://localhost:3001/calendar/holidays`
  - Body: `{ "country_code": "JP", "year": 2025 }`
- FX conversion (exchangerate.host)
  - POST `http://localhost:3001/fx/convert`
  - Body: `{ "amount": 200, "from": "USD", "to": "JPY" }`
  - Response: `{ "rate": <number>, "converted": <number> }`

Orchestrator (streams SSE):
- POST `http://localhost:3002/plan-day`
- Body: `{ "prompt": "Plan 10:00–18:00 in Kyoto on 2025-12-12. Prefer temples and walkable" }`
- Emits:
  - `tool_trace` entries for each tool call with durations
  - `plan_chunk` entries with Markdown segments
  - `[DONE]` when complete


## 3) Notes on Gemini prompts for routing and summarization

### Intent classification
System instruction:
- “You are an assistant that classifies user requests. Respond only with a JSON object.”
Model must produce:
```json
{
  "intent": "plan_day | refine_plan | compare_options",
  "city": "string",
  "date": "YYYY-MM-DD",
  "country_code": "2-letter ISO",
  "preferences": ["..."]
}
```
If classification is not `plan_day` but the prompt contains a currency conversion (e.g., “Convert 200 USD to JPY”), the orchestrator short-circuits to the FX tool and returns a small Markdown block with the rate and converted amount.


### Venue planning prompt
Inputs to the planner include:
- Weather summary, air-quality summary, holidays, preferences, basic geocoding
Planner returns a JSON array of 4–6 venue names or types.
The orchestrator then queries nearby places for each item and builds a point list for routing.


### Routing and ETA
When there are at least two points, the orchestrator calls the route ETA tool (`OSRM`) using a default profile (foot/bike/car). The result is included in the itinerary between stops.


### Summarization prompt
The final system instruction requires a Markdown itinerary and explicitly mandates:
1) Insert travel legs with ETAs between itinerary points  
2) If rain probability > 60%, add an indoor-heavy alternative section  
3) If PM2.5 > 75 μg/m³, append “mask recommended” to outdoor segments or propose indoor swaps  
4) Add a holiday caution when relevant  
5) Include a “Currency Conversion” section when FX is present  

For the final streaming output, the orchestrator uses a text-configured model instance to ensure Markdown text is returned (the classification model is configured for JSON).


## Troubleshooting
- “This demo only supports plan_day intent.”  
  Use a plan prompt, or include a currency expression like “convert 200 USD to JPY” for FX-only.
- Air data absent for a future date  
  OpenAQ is observational only. The air tool automatically drops future dates and fetches latest measurements near the coordinates; it also includes an Open‑Meteo fallback.
- No FX section in the output  
  Ensure your prompt contains a recognizable conversion phrase (e.g., “convert 200 USD to JPY”) and that `EXCHANGERATE_API_KEY` is set if your account requires it.
- 401/403 from OpenAQ  
  Ensure `OPENAQ_API_KEY` is valid in `.env` and visible in the container.


## Example prompts
- “Plan a museum-first day in Amsterdam on 2025-11-22. Bike preferred.”  
- “Refine: add a specialty coffee stop near the second venue.”  
- “Compare two options if it rains after 3pm.”  
- “Plan a day in Tokyo on 2025-11-20. Also convert 200 USD to JPY.”

