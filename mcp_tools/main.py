import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
import uvicorn
from mcp_tools.routers.geo import router as geo_router
from mcp_tools.routers.weather import router as weather_router
from mcp_tools.routers.air import router as air_router
from mcp_tools.routers.route import router as route_router
from mcp_tools.routers.calendar import router as calendar_router
from mcp_tools.routers.fx import router as fx_router


def get_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    expected_api_key = os.getenv("MCP_API_KEY")
    if expected_api_key is None or x_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

app = FastAPI(title="mcp_tools")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.include_router(geo_router, prefix="/geo")
app.include_router(weather_router, prefix="/weather")
app.include_router(air_router, prefix="/air")
app.include_router(route_router, prefix="/route")
app.include_router(calendar_router, prefix="/calendar")
app.include_router(fx_router, prefix="/fx")


@app.get("/", dependencies=[Depends(get_api_key)])
async def root(_: Request):
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001)

