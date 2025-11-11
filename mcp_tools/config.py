import os
from typing import Final


class _Config:
    def __init__(self) -> None:
        # Security / limits
        self.api_key: str | None = os.getenv("MCP_API_KEY")
        self.rate_limit: str = os.getenv("MCP_RATE_LIMIT", "30/minute")

        # HTTP behavior
        self.user_agent: str = os.getenv("MCP_USER_AGENT", "CityDayNavigator-MCP-Tool")
        try:
            self.http_timeout_sec: float = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))
        except ValueError:
            self.http_timeout_sec = 10.0

        # External API bases
        self.nominatim_base: str = os.getenv("NOMINATIM_BASE", "https://nominatim.openstreetmap.org")
        self.open_meteo_base: str = os.getenv("OPEN_METEO_BASE", "https://api.open-meteo.com/v1/forecast")
        self.openaq_base: str = os.getenv("OPENAQ_BASE", "https://api.openaq.org/v2")
        self.osrm_base: str = os.getenv("OSRM_BASE", "https://router.project-osrm.org")
        self.nager_base: str = os.getenv("NAGER_BASE", "https://date.nager.at/api/v3")
        self.fx_base: str = os.getenv("FX_BASE", "https://api.exchangerate.host")


CONFIG: Final[_Config] = _Config()

