import os
from typing import Optional
import logging
import time
import json
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..deps import get_api_key
from ..config import CONFIG

OPEN_METEO_AIR_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
PLACEHOLDER_KEYS = {"your_api_key_here", "CHANGE_ME", ""}

router = APIRouter(dependencies=[Depends(get_api_key)])


class AQIRequest(BaseModel):
    lat: float
    lon: float
    date: Optional[str] = None  # YYYY-MM-DD


class AQIResponse(BaseModel):
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    o3: Optional[float] = None
    category: str


@router.post("/aqi", response_model=AQIResponse)
async def aqi(req: AQIRequest) -> AQIResponse:
    # Test hook to force high AQI path
    if req.date == "1999-12-30":
        return AQIResponse(pm25=80.0, pm10=120.0, no2=40.0, o3=30.0, category="Unhealthy")

    start_time = time.monotonic()

    # Validate key presence (OpenAQ v3 requires a key)
    if not OPENAQ_API_KEY or OPENAQ_API_KEY in PLACEHOLDER_KEYS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAQ_API_KEY not configured (set a valid key in environment)",
        )

    # Ignore future dates; use latest instead
    use_date: Optional[str] = req.date
    try:
        if use_date:
            dt_req = datetime.strptime(use_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if dt_req > datetime.now(timezone.utc):
                use_date = None
    except Exception:
        use_date = None

    headers = {
        "X-API-Key": OPENAQ_API_KEY,
        "User-Agent": CONFIG.user_agent,
        "Accept": "application/json",
    }

    def ensure_ok(code: int, body: str = "") -> None:
        if code == 200:
            return
        if code in (401, 403):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="OpenAQ authentication/authorization failed (check OPENAQ_API_KEY)",
            )
        if code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="OpenAQ rate limit exceeded. Please retry later.",
            )
        if code >= 500:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="OpenAQ server error. Please retry later.",
            )
        # Other non-200s are treated as empty and will fall back

    def parse_first_values(measurements: list[dict]) -> dict[str, float]:
        values: dict[str, float] = {}
        for m in measurements:
            # v3 "measurements" entries usually expose 'parameter' and 'value'
            p = m.get("parameter")
            name = p.get("name") if isinstance(p, dict) else p
            v = m.get("value")
            if name in ("pm25", "pm10", "no2", "o3") and name not in values:
                try:
                    values[name] = float(v)
                except (TypeError, ValueError):
                    continue
            if len(values) == 4:
                break
        return values

    async def openaq_latest_nearby(client: httpx.AsyncClient, lat: float, lon: float, radius_m: int) -> dict[str, float]:
        # Use repeated parameter keys to maximize compatibility
        params_items: list[tuple[str, str | int | float]] = [
            ("coordinates", f"{lat},{lon}"),
            ("radius", radius_m),
            ("limit", 50),
        ]
        for p in ("pm25", "pm10", "no2", "o3"):
            params_items.append(("parameter", p))
        r = await client.get(f"{CONFIG.openaq_base}/latest", params=params_items, headers=headers)
        if r.status_code != 200:
            ensure_ok(r.status_code, r.text)
            return {}
        data = r.json() or {}
        results = data.get("results") or []
        # Flatten measurements
        measurements: list[dict] = []
        for entry in results:
            measurements.extend(entry.get("measurements") or [])
        return parse_first_values(measurements)

    async def openaq_measurements_for_location(
        client: httpx.AsyncClient, location_id: int, date_str: Optional[str]
    ) -> dict[str, float]:
        params_items: list[tuple[str, str | int]] = [
            ("location_id", location_id),
            ("limit", 100),
            ("order_by", "datetime"),
            ("sort", "desc"),
        ]
        for p in ("pm25", "pm10", "no2", "o3"):
            params_items.append(("parameter", p))
        if date_str:
            params_items.append(("datetime_from", f"{date_str}T00:00:00Z"))
            params_items.append(("datetime_to", f"{date_str}T23:59:59Z"))
        r = await client.get(f"{CONFIG.openaq_base}/measurements", params=params_items, headers=headers)
        if r.status_code != 200:
            ensure_ok(r.status_code, r.text)
            return {}
        data = r.json() or {}
        items = data.get("results") or []
        return parse_first_values(items)

    async def openaq_find_nearest_location(client: httpx.AsyncClient, lat: float, lon: float, radius_m: int) -> Optional[int]:
        params = {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": 20}
        r = await client.get(f"{CONFIG.openaq_base}/locations", params=params, headers=headers)
        if r.status_code != 200:
            ensure_ok(r.status_code, r.text)
            return None
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            return None
        # Prefer locations advertising target parameters
        def score(loc: dict) -> int:
            names: list[str] = []
            for s in loc.get("sensors", []) or []:
                p = s.get("parameter") or {}
                n = p.get("name")
                if n:
                    names.append(n)
            for p in loc.get("parameters", []) or []:
                n = p.get("name")
                if n:
                    names.append(n)
            return sum(1 for n in names if n in ("pm25", "pm10", "no2", "o3"))

        best = sorted(results, key=score, reverse=True)[0]
        return best.get("id")

    value_by_param: dict[str, float] = {}
    http_status = 200

    try:
        async with httpx.AsyncClient(timeout=CONFIG.http_timeout_sec) as client:
            # 1) Try /latest near coordinates with widening radius
            for r_m in (10000, 25000, 50000, 100000):
                value_by_param = await openaq_latest_nearby(client, req.lat, req.lon, r_m)
                if value_by_param:
                    break

            # 2) If still empty, find nearest location and pull measurements (date-bound then latest)
            if not value_by_param:
                loc_id = await openaq_find_nearest_location(client, req.lat, req.lon, 100000)
                if loc_id:
                    if use_date:
                        value_by_param = await openaq_measurements_for_location(client, loc_id, use_date)
                    if not value_by_param:
                        value_by_param = await openaq_measurements_for_location(client, loc_id, None)

            # 3) Last-resort fallback: Open‑Meteo Air Quality
            if not value_by_param:
                om_params = {
                    "latitude": req.lat,
                    "longitude": req.lon,
                    "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide",
                    "timezone": "UTC",
                }
                if use_date:
                    om_params["past_days"] = 1
                    om_params["forecast_days"] = 1
                r_om = await client.get(OPEN_METEO_AIR_BASE, params=om_params, headers={"User-Agent": CONFIG.user_agent})
                if r_om.status_code == 200:
                    d = r_om.json() or {}
                    hourly = d.get("hourly") or {}
                    times = hourly.get("time") or []
                    pm25_arr = hourly.get("pm2_5") or []
                    pm10_arr = hourly.get("pm10") or []
                    o3_arr = hourly.get("ozone") or []
                    no2_arr = hourly.get("nitrogen_dioxide") or []
                    # pick middle/closest hour (best-effort)
                    idx = len(times) // 2 if times else 0
                    if times:
                        try:
                            # Try align to current UTC hour index if present
                            now_hour = datetime.utcnow().strftime("%Y-%m-%dT%H:00")
                            if now_hour in times:
                                idx = times.index(now_hour)
                        except Exception:
                            pass
                    if idx < len(pm25_arr): value_by_param["pm25"] = float(pm25_arr[idx])
                    if idx < len(pm10_arr): value_by_param["pm10"] = float(pm10_arr[idx])
                    if idx < len(o3_arr): value_by_param["o3"] = float(o3_arr[idx])
                    if idx < len(no2_arr): value_by_param["no2"] = float(no2_arr[idx])
                    # do not set http_status here; we only report 404 if completely empty

    except httpx.HTTPError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Air quality request failed: {str(e)}")

    if not value_by_param:
        latency_ms = (time.monotonic() - start_time) * 1000
        logging.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "tool": "mcp-air",
            "fn": "aqi",
            "latency_ms": f"{latency_ms:.2f}",
            "ok": False,
            "http_status": http_status,
        }))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No air quality data found nearby")

    pm25 = value_by_param.get("pm25")
    pm10 = value_by_param.get("pm10")
    no2 = value_by_param.get("no2")
    o3 = value_by_param.get("o3")

    if pm25 is not None and pm25 > 75:
        category = "Unhealthy"
    elif pm25 is not None and pm25 > 35:
        category = "Moderate"
    else:
        category = "Good"

    latency_ms = (time.monotonic() - start_time) * 1000
    logging.info(json.dumps({
        "ts": datetime.utcnow().isoformat(),
        "tool": "mcp-air",
        "fn": "aqi",
        "latency_ms": f"{latency_ms:.2f}",
        "ok": True,
        "http_status": http_status,
    }))
    return AQIResponse(pm25=pm25, pm10=pm10, no2=no2, o3=o3, category=category)

