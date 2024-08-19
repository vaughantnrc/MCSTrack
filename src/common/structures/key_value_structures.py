import abc
from pydantic import BaseModel, Field
from typing import Union


class KeyValueSimpleAbstract(BaseModel, abc.ABC):
    """
    Abstract class to represent a key-value pair.
    Intended use: Setting parameter over network, serialization through JSON.
    """
    parsable_type: str
    key: str = Field()


class KeyValueSimpleBool(KeyValueSimpleAbstract):
    parsable_type: str = Field(default="bool", const=True)
    value: bool = Field()


class KeyValueSimpleFloat(KeyValueSimpleAbstract):
    parsable_type: str = Field(default="float", const=True)
    value: float = Field()


class KeyValueSimpleInt(KeyValueSimpleAbstract):
    parsable_type: str = Field(default="int", const=True)
    value: int = Field()


class KeyValueSimpleString(KeyValueSimpleAbstract):
    parsable_type: str = Field(default="str", const=True)
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
    parsable_type: str = Field(default="bool", const=True)
    value: bool = Field()

    def to_simple(self) -> KeyValueSimpleBool:
        return KeyValueSimpleBool(key=self.key, value=self.value)


class KeyValueMetaEnum(KeyValueMetaAbstract):
    parsable_type: str = Field(default="enum", const=True)
    value: str = Field()
    allowable_values: list[str] = Field(default_factory=list)

    def to_simple(self) -> KeyValueSimpleString:
        return KeyValueSimpleString(key=self.key, value=self.value)


class KeyValueMetaFloat(KeyValueMetaAbstract):
    parsable_type: str = Field(default="float", const=True)
    value: float = Field()
    range_minimum: float = Field()
    range_maximum: float = Field()
    range_step: float = Field(default=1.0)
    digit_count: int = Field(default=2)

    def to_simple(self) -> KeyValueSimpleFloat:
        return KeyValueSimpleFloat(key=self.key, value=self.value)


class KeyValueMetaInt(KeyValueMetaAbstract):
    parsable_type: str = Field(default="int", const=True)
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
