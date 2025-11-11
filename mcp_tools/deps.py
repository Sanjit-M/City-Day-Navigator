from typing import Optional

from fastapi import Header, HTTPException, status

from .config import CONFIG


def get_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    expected_api_key = CONFIG.api_key
    if expected_api_key is None or x_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


