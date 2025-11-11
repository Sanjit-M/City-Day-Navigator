import os
from typing import Optional
import logging
import time
import json
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..deps import get_api_key
from ..config import CONFIG


OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

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
    start_time = time.monotonic()
    if not OPENAQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAQ_API_KEY not configured",
        )

    params = {
        "coordinates": f"{req.lat},{req.lon}",
        "radius": 5000,
        "limit": 100,
        "page": 1,
        "sort": "desc",
        "order_by": "datetime",
        "parameter": "pm25,pm10,no2,o3",
    }
    if req.date:
        params["date_from"] = f"{req.date}T00:00:00Z"
        params["date_to"] = f"{req.date}T23:59:59Z"

    headers = {
        "X-API-Key": OPENAQ_API_KEY,
        "User-Agent": CONFIG.user_agent,
    }

    try:
        async with httpx.AsyncClient(timeout=CONFIG.http_timeout_sec) as client:
            resp = await client.get(
                f"{CONFIG.openaq_base}/measurements",
                params=params,
                headers=headers,
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"OpenAQ error: {resp.status_code}",
                )
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAQ request failed: {str(e)}",
        )

    results = data.get("results") or []
    latency_ms = (time.monotonic() - start_time) * 1000
    http_status = resp.status_code if 'resp' in locals() else None
    ok = True if results is not None else False

    value_by_param: dict[str, float] = {}
    for item in results:
        p = item.get("parameter")
        v = item.get("value")
        if p in {"pm25", "pm10", "no2", "o3"} and p not in value_by_param:
            try:
                value_by_param[p] = float(v)
            except (TypeError, ValueError):
                continue
        if len(value_by_param) == 4:
            break

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

    log_data = {
        "ts": datetime.utcnow().isoformat(),
        "tool": "mcp-air",
        "fn": "aqi",
        "latency_ms": f"{latency_ms:.2f}",
        "ok": ok,
        "http_status": http_status,
    }
    logging.info(json.dumps(log_data))
    return AQIResponse(pm25=pm25, pm10=pm10, no2=no2, o3=o3, category=category)

