from typing import Dict, List, Optional
import logging
import time
import json
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..main import get_api_key


router = APIRouter(dependencies=[Depends(get_api_key)])


class GeocodeRequest(BaseModel):
    city: str
    country_hint: Optional[str] = None


class GeocodeResponse(BaseModel):
    lat: float
    lon: float
    display_name: str


@router.post("/geocode", response_model=GeocodeResponse)
async def geocode(req: GeocodeRequest) -> GeocodeResponse:
    start_time = time.monotonic()
    query = req.city if not req.country_hint else f"{req.city}, {req.country_hint}"
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 0,
    }
    headers = {"User-Agent": "CityDayNavigator-MCP-Tool"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
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
        display_name=first.get("display_name", ""),
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
    tags: Dict[str, str] = {}


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
async def nearby(req: NearbyRequest) -> NearbyResponse:
    start_time = time.monotonic()
    viewbox = _bbox_from_center(req.lat, req.lon, req.radius_m)
    params = {
        "q": req.query,
        "format": "jsonv2",
        "limit": req.limit,
        "addressdetails": 0,
        "viewbox": viewbox,
        "bounded": 1,
        "extratags": 1,
    }
    headers = {"User-Agent": "CityDayNavigator-MCP-Tool"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
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

    results: List[NearbyItem] = []
    for item in data:
        name = item.get("display_name") or item.get("name") or ""
        tags = item.get("extratags") or {}
        # Ensure tags keys/values are strings
        str_tags: Dict[str, str] = {str(k): str(v) for k, v in tags.items()}
        results.append(
            NearbyItem(
                name=name,
                lat=float(item["lat"]),
                lon=float(item["lon"]),
                tags=str_tags,
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

