from .image_processing import \
    Annotation
from .serialization import \
    KeyValueMetaAny, \
    KeyValueSimpleAny
from .status import \
    MCTError, \
    SeverityLabel, \
    StatusMessageSource
import abc
import datetime
from enum import StrEnum
import numpy
from pydantic import BaseModel, Field


class _Configuration(BaseModel):
    method: str = Field()


class _Status(StrEnum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    FAILURE = "FAILURE"


class MCTAnnotatorRuntimeError(MCTError):
    message: str

    def __init__(self, message: str, *args):
        super().__init__(args)
        self.message = message


class Annotator(abc.ABC):
    """
    Functions may raise MCTAnnotatorRuntimeError
    """

    Configuration: type[_Configuration] = _Configuration
    Status: type[_Status] = _Status

    _configuration: Configuration
    _status: Status
    _status_message_source: StatusMessageSource

    def __init__(
        self,
        configuration: Configuration,
        status_message_source: StatusMessageSource
    ):
        self._configuration = configuration
        self._status_message_source = status_message_source

    def add_status_message(
        self,
        severity: SeverityLabel,
        message: str
    ) -> None:
        self._status_message_source.enqueue_status_message(severity=severity, message=message)

    def get_status(self) -> Status:
        return self._status

    def set_status(self, status: Status) -> None:
        self._status = status

    @abc.abstractmethod
    def get_changed_timestamp(self) -> datetime.datetime: ...

    @abc.abstractmethod
    def get_markers_detected(self) -> list[Annotation]: ...

    @abc.abstractmethod
    def get_markers_rejected(self) -> list[Annotation]: ...

    @abc.abstractmethod
    def get_parameters(self) -> list[KeyValueMetaAny]: ...

    @staticmethod
    @abc.abstractmethod
    def get_type_identifier() -> str: ...  # Unique string associated with this type

    @abc.abstractmethod
    def set_parameters(self, parameters: list[KeyValueSimpleAny]) -> None: ...

    @abc.abstractmethod
    def update(self, image: numpy.ndarray) -> None: ...
