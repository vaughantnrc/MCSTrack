from typing import TypeVar


T = TypeVar("T")


class PythonUtils:
    """
    A "class" to group related static functions, like in a namespace.
    The class itself is not meant to be instantiated.
    """

    def __init__(self):
        raise RuntimeError(f"{__class__.__name__} is not meant to be instantiated.")

    @staticmethod
    def get_kwarg(
            kwargs: dict,
            key: str,
            arg_type: type[T],
            required: bool = True
    ) -> T | None:
        """
        :param kwargs: kwargs as a dict (without the "**")
        :param key: key to search for
        :param arg_type: expected type
        :param required: If the keyword does not exist, then:
           required == True -> Raise ValueError
           required == False -> Return None
        """

        if key not in kwargs:
            if required:
                raise ValueError(f"Missing required key {key} in keyword arguments.")
            return None
        value: T = kwargs[key]
        if not isinstance(value, arg_type):
            raise ValueError(
                f"Expected keyword argument {key} to be of type {arg_type.__name__}, "
                f"but got {type(value).__name__}.")
        return value

