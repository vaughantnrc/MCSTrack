import hjson
import json
import os
from typing import Any, Callable, Literal


class IOUtils:
    """
    static class for IO-related utility functions.
    """

    PathType = Literal["filepath", "path"]

    @staticmethod
    def exists(
        path: str,
        pathtype: PathType,
        on_error_for_user: Callable[[str], Any],
        on_error_for_dev: Callable[[str], Any],
        create_path: bool = False
    ) -> bool:
        """
        :param path: Location of target
        :param pathtype: "filepath" (target is file) or "path" (target is directory)
        :param on_error_for_user:
            Function to supply with a publicly-viewable string (message viewable by end-user) in the event of an error.
        :param on_error_for_dev:
            Function to supply with a developer-friendly string (not by end-user) in the event of an error.
        :param create_path:
            If pathtype == "path" and this is True, try to create the path if it does not exist. Default False.
        :return: True if there is a file/path at the indicated path, otherwise False.
        """
        exists: bool = os.path.exists(path)
        if pathtype == "filepath":
            if exists and not os.path.isfile(path):
                on_error_for_user(
                    "Filepath location exists but is not a file. "
                    "Most likely a directory exists at that location, "
                    "and it needs to be manually removed.")
                on_error_for_dev(
                    f"Specified filepath location {path} exists but is not a file.")
                return False
        elif pathtype == "path":
            if exists and not os.path.isdir(path):
                on_error_for_user(
                    "Path location exists but is not a path. "
                    "Most likely a file exists at that location, "
                    "and it needs to be manually removed.")
                on_error_for_dev(
                    f"Specified path location {path} exists but is not a path.")
                return False
            if create_path and not exists:
                os.makedirs(name=path, exist_ok=True)
                exists = os.path.exists(path)
        return exists

    @staticmethod
    def hjson_read(
        filepath: str,
        on_error_for_user: Callable[[str], Any],
        on_error_for_dev: Callable[[str], Any]
    ) -> dict | None:
        """
        :param filepath:
        :param on_error_for_user:
            Function to supply with a publicly-viewable string (message viewable by end-user) in the event of an error.
        :param on_error_for_dev:
            Function to supply with a developer-friendly string (not by end-user) in the event of an error.
        :return: Dictionary representing the JSON data if successful, otherwise None
        """
        if not IOUtils.exists(
            path=filepath,
            pathtype="filepath",
            on_error_for_user=on_error_for_user,
            on_error_for_dev=on_error_for_dev
        ):
            return None
        json_dict: dict
        try:
            with open(filepath, 'r', encoding='utf-8') as input_file:
                json_dict = hjson.load(input_file)
        except OSError as e:
            on_error_for_user("An unexpected file I/O error happened while reading a file.")
            on_error_for_dev(str(e))
            return None
        return json_dict

    @staticmethod
    def json_read(
        filepath: str,
        on_error_for_user: Callable[[str], Any],
        on_error_for_dev: Callable[[str], Any]
    ) -> dict | None:
        """
        Generally it is better to use hjson_read instead, which is less strict and supports comments.
        This function currently forwards to hjson_ready anyway.
        """
        return IOUtils.hjson_read(
            filepath=filepath,
            on_error_for_user=on_error_for_user,
            on_error_for_dev=on_error_for_dev)

    @staticmethod
    def json_write(
        filepath: str,
        json_dict: dict,
        on_error_for_user: Callable[[str], Any],
        on_error_for_dev: Callable[[str], Any],
        ignore_none: bool = False,
        indent: int = 4
    ) -> bool:
        """
        :param filepath:
        :param json_dict:
        :param on_error_for_user:
            Function to supply with a publicly-viewable string (message viewable by end-user) in the event of an error.
        :param on_error_for_dev:
            Function to supply with a developer-friendly string (not by end-user) in the event of an error.
        :param ignore_none: Explicitly ignore (don't write) any key-value pairs where the value is None. Default False.
        :param indent: Width (spaces) of each indentation level. Default 4.
        :return: True if the file was written, otherwise False.
        """
        filepath = os.path.join(filepath)
        path = os.path.dirname(filepath)
        if not IOUtils.exists(
            path=path,
            pathtype="path",
            on_error_for_user=on_error_for_user,
            on_error_for_dev=on_error_for_dev,
            create_path=True
        ):
            return False
        if ignore_none:
            json_dict = IOUtils._remove_all_none_from_dict_recursive(json_dict)
        try:
            with open(filepath, 'w', encoding='utf-8') as output_file:
                json.dump(json_dict, output_file, sort_keys=False, indent=indent)
        except OSError as e:
            on_error_for_user("An unexpected file I/O error happened while writing a file.")
            on_error_for_dev(str(e))
            return False
        return True

    @staticmethod
    def _remove_all_none_from_dict_recursive(
        input_dict: dict
    ) -> dict:
        output_dict = dict(input_dict)
        for key in output_dict.keys():
            if isinstance(output_dict[key], dict):
                output_dict[key] = IOUtils._remove_all_none_from_dict_recursive(output_dict[key])
            elif output_dict[key] is None:
                del output_dict[key]
        return output_dict
