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

    from_ccy = req.from_currency.upper()
    to_ccy = req.to_currency.upper()

    headers = {"User-Agent": CONFIG.user_agent}

    try:
        async with httpx.AsyncClient(timeout=CONFIG.http_timeout_sec) as client:
            # First attempt: use /convert
            convert_params = {
                "from": from_ccy,
                "to": to_ccy,
                "amount": req.amount,
            }
            if EXCHANGERATE_API_KEY:
                convert_params["access_key"] = EXCHANGERATE_API_KEY

            resp = await client.get(
                f"{CONFIG.fx_base}/convert",
                params=convert_params,
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Some deployments include a success flag and error info
                if isinstance(data, dict) and data.get("success") is False:
                    # fall through to fallback
                    pass
                else:
                    converted_raw = data.get("result")
                    info = data.get("info") or {}
                    # Accept either 'rate' or 'quote' per docs variance
                    rate_raw = info.get("rate", info.get("quote"))
                    if converted_raw is not None and rate_raw is not None:
                        rate_f = float(rate_raw)
                        converted_f = float(converted_raw)
                        latency_ms = (time.monotonic() - start_time) * 1000
                        http_status = resp.status_code
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
                        return ConvertResponse(rate=rate_f, converted=converted_f)

            # Fallback: /latest with manual multiplication
            latest_params = {"base": from_ccy, "symbols": to_ccy}
            if EXCHANGERATE_API_KEY:
                latest_params["access_key"] = EXCHANGERATE_API_KEY
            resp2 = await client.get(
                f"{CONFIG.fx_base}/latest",
                params=latest_params,
                headers=headers,
            )
            if resp2.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"exchangerate.host error: {resp2.status_code}",
                )
            data2 = resp2.json()
            rates = (data2.get("rates") or {}) if isinstance(data2, dict) else {}
            rate = rates.get(to_ccy)
            if rate is None:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="exchangerate.host response missing requested symbol rate",
                )
            try:
                rate_f = float(rate)
                converted_f = float(req.amount) * rate_f
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="exchangerate.host response malformed",
                )

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"exchangerate.host request failed: {str(e)}",
        )

    latency_ms = (time.monotonic() - start_time) * 1000
    http_status = resp2.status_code if 'resp2' in locals() else None
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
    return ConvertResponse(rate=rate_f, converted=converted_f)

