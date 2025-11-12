from typing import Dict, List, Optional
import logging
import time
import json
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..deps import get_api_key, get_http_client
from ..config import CONFIG


router = APIRouter(dependencies=[Depends(get_api_key)])


class GeocodeRequest(BaseModel):
    city: str
    country_hint: Optional[str] = None


class GeocodeResponse(BaseModel):
    lat: float
    lon: float


@router.post("/geocode", response_model=GeocodeResponse)
async def geocode(req: GeocodeRequest, client: httpx.AsyncClient = Depends(get_http_client)) -> GeocodeResponse:
    start_time = time.monotonic()
    query = req.city if not req.country_hint else f"{req.city}, {req.country_hint}"
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 0,
    }
    headers = {"User-Agent": CONFIG.user_agent}
    try:
        resp = await client.get(
            f"{CONFIG.nominatim_base}/search",
            params=params,
            headers=headers,
        )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Nominatim error: {resp.status_code}",
            )
        data = resp.json()
        latency_ms = (time.monotonic() - start_time) * 1000
        http_status = resp.status_code
        ok = True
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Nominatim request failed: {str(e)}",
        )

    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No results")

    first = data[0]
    log_data = {
        "ts": datetime.utcnow().isoformat(),
        "tool": "mcp-geo",
        "fn": "geocode",
        "latency_ms": f"{latency_ms:.2f}",
        "ok": ok,
        "http_status": http_status,
    }
    logging.info(json.dumps(log_data))
    return GeocodeResponse(
        lat=float(first["lat"]),
        lon=float(first["lon"]),
    )


class NearbyRequest(BaseModel):
    lat: float
    lon: float
    query: str
    radius_m: int = Field(..., le=5000)
    limit: int = Field(..., le=15)


class NearbyItem(BaseModel):
    name: str
    lat: float
    lon: float


class NearbyResponse(BaseModel):
    results: List[NearbyItem]


def _bbox_from_center(lat: float, lon: float, radius_m: int) -> str:
    # Approximate degree deltas from meters
    # 1 deg lat ~ 111.32 km; 1 deg lon ~ 111.32 km * cos(lat)
    import math

    delta_lat = radius_m / 111_320.0
    delta_lon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    left = lon - delta_lon
    right = lon + delta_lon
    top = lat + delta_lat
    bottom = lat - delta_lat
    # Nominatim expects "left,top,right,bottom"
    return f"{left},{top},{right},{bottom}"


@router.post("/nearby", response_model=NearbyResponse)
async def nearby(req: NearbyRequest, client: httpx.AsyncClient = Depends(get_http_client)) -> NearbyResponse:
    start_time = time.monotonic()
    viewbox = _bbox_from_center(req.lat, req.lon, req.radius_m)
    params = {
        "q": req.query,
        "format": "jsonv2",
        "limit": req.limit,
        "addressdetails": 0,
        "viewbox": viewbox,
        "bounded": 1,
    }
    headers = {"User-Agent": CONFIG.user_agent}
    try:
        resp = await client.get(
            f"{CONFIG.nominatim_base}/search",
            params=params,
            headers=headers,
        )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Nominatim error: {resp.status_code}",
            )
        data = resp.json()
        latency_ms = (time.monotonic() - start_time) * 1000
        http_status = resp.status_code
        ok = True
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Nominatim request failed: {str(e)}",
        )

    # Deterministic ranking: distance from center, then name
    def _dist2(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
        # Approximate squared distance accounting for longitude shrink by latitude
        import math
        dlat = a_lat - b_lat
        dlon = (a_lon - b_lon) * math.cos(math.radians((a_lat + b_lat) / 2.0))
        return dlat * dlat + dlon * dlon

    enriched: List[Dict[str, object]] = []
    for item in data:
        try:
            i_lat = float(item["lat"])
            i_lon = float(item["lon"])
        except (TypeError, ValueError, KeyError):
            continue
        name = item.get("display_name") or item.get("name") or ""
        dist2 = _dist2(req.lat, req.lon, i_lat, i_lon)
        enriched.append({"name": str(name), "lat": i_lat, "lon": i_lon, "dist2": dist2})

    enriched.sort(key=lambda x: (x["dist2"], str(x["name"]).lower()))
    # Trim to requested limit after sorting (in case API returned more)
    enriched = enriched[: req.limit]

    results: List[NearbyItem] = []
    for e in enriched:
        results.append(
            NearbyItem(
                name=str(e["name"]),
                lat=float(e["lat"]),
                lon=float(e["lon"]),
            )
        )
    log_data = {
        "ts": datetime.utcnow().isoformat(),
        "tool": "mcp-geo",
        "fn": "nearby",
        "latency_ms": f"{latency_ms:.2f}",
        "ok": ok,
        "http_status": http_status,
    }
    logging.info(json.dumps(log_data))
    return NearbyResponse(results=results)

