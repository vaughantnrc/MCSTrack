import logging
from pydantic import BaseModel, Field
from typing import Final, Literal


SEVERITY_LABEL_DEBUG: Final[str] = "debug"
SEVERITY_LABEL_INFO: Final[str] = "info"
SEVERITY_LABEL_WARNING: Final[str] = "warning"
SEVERITY_LABEL_ERROR: Final[str] = "error"
SEVERITY_LABEL_CRITICAL: Final[str] = "critical"

SeverityLabel = Literal["debug", "info", "warning", "error", "critical"]
SEVERITY_LABEL_TO_INT: Final[dict[SeverityLabel, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL}


class StatusMessage(BaseModel):
    source_label: str = Field()
    severity: SeverityLabel = Field()
    message: str
    timestamp_utc_iso8601: str
