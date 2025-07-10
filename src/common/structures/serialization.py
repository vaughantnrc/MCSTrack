from src.common.exceptions import MCTError
import abc
from pydantic import BaseModel, Field, ValidationError
from typing import Final, Literal, TypeVar, Union


class KeyValueSimpleAbstract(BaseModel, abc.ABC):
    """
    Abstract class to represent a key-value pair.
    Intended use: Setting parameter over network, serialization through JSON.
    """
    parsable_type: str
    key: str = Field()


class KeyValueSimpleBool(KeyValueSimpleAbstract):
    _TYPE_IDENTIFIER: Final[str] = "bool"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: bool = Field()


class KeyValueSimpleFloat(KeyValueSimpleAbstract):
    _TYPE_IDENTIFIER: Final[str] = "float"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: float = Field()


class KeyValueSimpleInt(KeyValueSimpleAbstract):
    _TYPE_IDENTIFIER: Final[str] = "int"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: int = Field()


class KeyValueSimpleString(KeyValueSimpleAbstract):
    _TYPE_IDENTIFIER: Final[str] = "str"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: str = Field()


class KeyValueMetaAbstract(BaseModel, abc.ABC):
    """
    Abstract class to represent a key-value pair, and additional data about the datum (range, description, etc)
    Intended use: Getting parameter over network, serialization through JSON.
    """
    parsable_type: str
    key: str = Field()

    @abc.abstractmethod
    def to_simple(self) -> KeyValueSimpleAbstract: ...


class KeyValueMetaBool(KeyValueMetaAbstract):
    _TYPE_IDENTIFIER: Final[str] = "bool"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: bool = Field()

    def to_simple(self) -> KeyValueSimpleBool:
        return KeyValueSimpleBool(key=self.key, value=self.value)


class KeyValueMetaEnum(KeyValueMetaAbstract):
    _TYPE_IDENTIFIER: Final[str] = "enum"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: str = Field()
    allowable_values: list[str] = Field(default_factory=list)

    def to_simple(self) -> KeyValueSimpleString:
        return KeyValueSimpleString(key=self.key, value=self.value)


class KeyValueMetaFloat(KeyValueMetaAbstract):
    _TYPE_IDENTIFIER: Final[str] = "float"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: float = Field()
    range_minimum: float = Field()
    range_maximum: float = Field()
    range_step: float = Field(default=1.0)
    digit_count: int = Field(default=2)

    def to_simple(self) -> KeyValueSimpleFloat:
        return KeyValueSimpleFloat(key=self.key, value=self.value)


class KeyValueMetaInt(KeyValueMetaAbstract):
    _TYPE_IDENTIFIER: Final[str] = "int"

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    value: int = Field()
    range_minimum: int = Field()
    range_maximum: int = Field()
    range_step: int = Field(default=1)

    def to_simple(self) -> KeyValueSimpleInt:
        return KeyValueSimpleInt(key=self.key, value=self.value)


# pydantic doesn't appear to handle very well typing's (TypeA, TypeB, ...) notation of a union
KeyValueSimpleAny = Union[
    KeyValueSimpleBool,
    KeyValueSimpleString,
    KeyValueSimpleFloat,
    KeyValueSimpleInt]
KeyValueMetaAny = Union[
    KeyValueMetaBool,
    KeyValueMetaEnum,
    KeyValueMetaFloat,
    KeyValueMetaInt]


def key_value_meta_to_simple(
    key_value_meta_list: list[KeyValueMetaAny]
) -> list[KeyValueSimpleAny]:
    return [key_value_meta.to_simple() for key_value_meta in key_value_meta_list]


DeserializableT = TypeVar('DeserializableT', bound='MCTParsable')


class MCTSerializationError(MCTError):
    message: str

    def __init__(self, message: str, *args):
        super().__init__(args)
        self.message = message


class MCTDeserializable(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type_identifier() -> str:
        pass

    @staticmethod
    def deserialize_series_list(
        series_dict: dict,
        supported_types: list[type[DeserializableT]]
    ) -> list[DeserializableT]:
        if "series" not in series_dict or not isinstance(series_dict["series"], list):
            message: str = "parsable_series_dict did not contain field series. Input is improperly formatted."
            raise MCTSerializationError(message)

        output_series: list[DeserializableT] = list()
        for parsable_dict in series_dict["series"]:
            if not isinstance(parsable_dict, dict):
                message: str = "series contained a non-dict element. Input is improperly formatted."
                raise MCTSerializationError(message)
            output_series.append(MCTDeserializable.deserialize_single(
                single_dict=parsable_dict,
                supported_types=supported_types))

        return output_series

    @staticmethod
    def deserialize_single(
        single_dict: dict,
        supported_types: list[type[DeserializableT]]
    ) -> DeserializableT:
        if "parsable_type" not in single_dict or not isinstance(single_dict["parsable_type"], str):
            message: str = "parsable_dict did not contain parsable_type. Input is improperly formatted."
            raise MCTSerializationError(message) from None

        for supported_type in supported_types:
            if single_dict["parsable_type"] == supported_type.parsable_type_identifier():
                request: DeserializableT
                try:
                    request = supported_type(**single_dict)
                except ValidationError as e:
                    raise MCTSerializationError(f"A request of type {supported_type} was ill-formed: {str(e)}") from None
                return request

        message: str = "parsable_type did not match any expected value. Input is improperly formatted."
        raise MCTSerializationError(message)
