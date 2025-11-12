from typing import Optional
import httpx

from fastapi import Header, HTTPException, status, Request

from .config import CONFIG


def get_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    expected_api_key = CONFIG.api_key
    if expected_api_key is None or x_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


def get_http_client(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "http_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="HTTP client not initialized",
        )
    return client

