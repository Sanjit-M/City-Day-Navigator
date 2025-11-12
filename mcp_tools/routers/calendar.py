from typing import List
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


class HolidaysRequest(BaseModel):
    country_code: str
    year: int


class Holiday(BaseModel):
    date: str
    localName: str


class HolidaysResponse(BaseModel):
    holidays: List[Holiday]


@router.post("/holidays", response_model=HolidaysResponse)
async def holidays(req: HolidaysRequest, client: httpx.AsyncClient = Depends(get_http_client)) -> HolidaysResponse:
    start_time = time.monotonic()
    url = f"{CONFIG.nager_base}/PublicHolidays/{req.year}/{req.country_code}"
    headers = {"User-Agent": CONFIG.user_agent}

    try:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Nager.Date error: {resp.status_code}",
            )
        data = resp.json()
        latency_ms = (time.monotonic() - start_time) * 1000
        http_status = resp.status_code
        ok = True
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Nager.Date request failed: {str(e)}",
        )

    holidays_list: List[Holiday] = []
    if isinstance(data, list):
        for item in data:
            holidays_list.append(
                Holiday(
                    date=str(item.get("date", "")),
                    localName=str(item.get("localName", "")),
                )
            )

    log_data = {
        "ts": datetime.utcnow().isoformat(),
        "tool": "mcp-calendar",
        "fn": "holidays",
        "latency_ms": f"{latency_ms:.2f}",
        "ok": ok,
        "http_status": http_status,
    }
    logging.info(json.dumps(log_data))
    return HolidaysResponse(holidays=holidays_list)

