import abc


class MCastParsable(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def parsable_type_identifier() -> str:
        pass
