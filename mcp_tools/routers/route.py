from typing import List, Literal, Optional
import logging
import time
import json
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..deps import get_api_key, get_http_client
from ..config import CONFIG


router = APIRouter(dependencies=[Depends(get_api_key)])


class Point(BaseModel):
    lat: float
    lon: float


class ETARequest(BaseModel):
    points: List[Point]
    profile: Literal["foot", "bike", "car"]


class ETAResponse(BaseModel):
    distance_km: float
    duration_min: float


@router.post("/eta", response_model=ETAResponse)
async def eta(req: ETARequest, client: httpx.AsyncClient = Depends(get_http_client)) -> ETAResponse:
    start_time = time.monotonic()
    if not req.points or len(req.points) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least two points are required",
        )

    coords = ";".join(f"{p.lon},{p.lat}" for p in req.points)
    url = f"{CONFIG.osrm_base}/route/v1/{req.profile}/{coords}"
    params = {
        # No geometry to minimize payload size
        "overview": "false",
    }
    headers = {"User-Agent": CONFIG.user_agent}

    try:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OSRM error: {resp.status_code}",
            )
        data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OSRM request failed: {str(e)}",
        )

    routes = data.get("routes") or []
    if not routes:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OSRM returned no routes",
        )

    route0 = routes[0]
    try:
        distance_km = float(route0.get("distance", 0.0)) / 1000.0
        duration_min = float(route0.get("duration", 0.0)) / 60.0
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OSRM response malformed",
        )

    latency_ms = (time.monotonic() - start_time) * 1000
    http_status = resp.status_code if 'resp' in locals() else None
    ok = True
    log_data = {
        "ts": datetime.utcnow().isoformat(),
        "tool": "mcp-route",
        "fn": "eta",
        "latency_ms": f"{latency_ms:.2f}",
        "ok": ok,
        "http_status": http_status,
    }
    logging.info(json.dumps(log_data))
    return ETAResponse(distance_km=distance_km, duration_min=duration_min)

