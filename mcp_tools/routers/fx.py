import os
from typing import Optional
import logging
import time
import json
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..deps import get_api_key
from ..config import CONFIG


EXCHANGERATE_API_KEY = os.getenv("EXCHANGERATE_API_KEY")

router = APIRouter(dependencies=[Depends(get_api_key)])


class ConvertRequest(BaseModel):
    amount: float
    from_currency: str = Field(..., alias="from")
    to_currency: str = Field(..., alias="to")

    class Config:
        populate_by_name = True


class ConvertResponse(BaseModel):
    rate: float
    converted: float


@router.post("/convert", response_model=ConvertResponse)
async def convert(req: ConvertRequest) -> ConvertResponse:
    start_time = time.monotonic()
    params = {
        "base": req.from_currency,
        "symbols": req.to_currency,
    }
    if EXCHANGERATE_API_KEY:
        params["access_key"] = EXCHANGERATE_API_KEY

    headers = {"User-Agent": CONFIG.user_agent}

    try:
        async with httpx.AsyncClient(timeout=CONFIG.http_timeout_sec) as client:
            resp = await client.get(
                f"{CONFIG.fx_base}/latest",
                params=params,
                headers=headers,
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"exchangerate.host error: {resp.status_code}",
                )
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"exchangerate.host request failed: {str(e)}",
        )

    rates = data.get("rates") or {}
    rate = rates.get(req.to_currency.upper())
    if rate is None:
        # some responses might use lowercase or mismatched case
        rate = rates.get(req.to_currency)
    if rate is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="exchangerate.host response missing requested symbol rate",
        )

    try:
        rate_f = float(rate)
        converted = float(req.amount) * rate_f
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="exchangerate.host response malformed",
        )

    latency_ms = (time.monotonic() - start_time) * 1000
    http_status = resp.status_code if 'resp' in locals() else None
    ok = True
    log_data = {
        "ts": datetime.utcnow().isoformat(),
        "tool": "mcp-fx",
        "fn": "convert",
        "latency_ms": f"{latency_ms:.2f}",
        "ok": ok,
        "http_status": http_status,
    }
    logging.info(json.dumps(log_data))
    return ConvertResponse(rate=rate_f, converted=converted)

