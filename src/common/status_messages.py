import datetime
from enum import StrEnum
import logging
from pydantic import BaseModel, Field
from typing import Final


logger = logging.getLogger(__name__)


class SeverityLabel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


SEVERITY_LABEL_TO_INT: Final[dict[SeverityLabel, int]] = {
    SeverityLabel.DEBUG: logging.DEBUG,
    SeverityLabel.INFO: logging.INFO,
    SeverityLabel.WARNING: logging.WARNING,
    SeverityLabel.ERROR: logging.ERROR,
    SeverityLabel.CRITICAL: logging.CRITICAL}


class StatusMessage(BaseModel):
    source_label: str = Field()
    severity: SeverityLabel = Field()
    message: str
    timestamp_utc_iso8601: str


class StatusMessageSource:
    """
    Class to facilitate the management of status messages sent between components
    """

    _source_label: str

    # send to other system components (for e.g. viewing, retransmission, etc)
    _status_message_outboxes: dict[str, list[StatusMessage]]

    _send_to_logger: bool

    def __init__(
        self,
        source_label,
        send_to_logger: bool = True
    ):
        self._status_message_outboxes = dict()
        self._source_label = source_label
        self._send_to_logger = send_to_logger

    def add_status_subscriber(
        self,
        subscriber_label: str
    ):
        if subscriber_label not in self._status_message_outboxes:
            self._status_message_outboxes[subscriber_label] = list()
            message: str = f"{subscriber_label} is now listening for status messages."
            self.enqueue_status_message(
                source_label=self._source_label,
                severity=SeverityLabel.DEBUG,
                message=message)
        else:
            message: str = f"{subscriber_label} is already in status message outboxes."
            self.enqueue_status_message(
                source_label=self._source_label,
                severity=SeverityLabel.ERROR,
                message=message)

    def get_source_label(self) -> str:
        return self._source_label

    def enqueue_status_message(
        self,
        severity: SeverityLabel,
        message: str,
        source_label: str | None = None,
        timestamp_utc_iso8601: datetime.datetime | str | None = None
    ):
        if not source_label:
            source_label = self._source_label
        if not timestamp_utc_iso8601:
            timestamp_utc_iso8601 = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        elif isinstance(timestamp_utc_iso8601, datetime.datetime):
            timestamp_utc_iso8601 = timestamp_utc_iso8601.isoformat()
        message: StatusMessage = StatusMessage(
            source_label=source_label,
            severity=severity,
            message=message,
            timestamp_utc_iso8601=timestamp_utc_iso8601)
        if self._send_to_logger:
            # In hindsight, it might be a good idea to look at the built-in
            # logger's functionalities and see if we really need this class
            if severity == "debug":
                logger.debug(message)
            elif severity == "info":
                logger.info(message)
            elif severity == "warning":
                logger.warning(message)
            elif severity == "error":
                logger.error(message)
            elif severity == "critical":
                logger.critical(message)
            else:
                logger.exception(
                    f"Unhandled status severity {severity} "
                    f"for message {message}.")
        for outbox in self._status_message_outboxes.values():
            outbox.append(message)

    def pop_new_status_messages(
        self,
        subscriber_label: str
    ) -> list[StatusMessage]:
        if subscriber_label not in self._status_message_outboxes:
            raise RuntimeError(
                f"subscriber_label {subscriber_label} not found - cannot retrieve status messages.")
        status_messages: list[StatusMessage] = list(self._status_message_outboxes[subscriber_label])
        self._status_message_outboxes[subscriber_label].clear()
        return status_messages
