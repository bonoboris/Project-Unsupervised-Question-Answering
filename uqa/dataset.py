"""
Classes to load and save data.
"""

import abc
import copy
import errno
import logging
import os
import pickle
import random as rd
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

import ujson as json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TJson = Union[Dict, List]
DataIterable = Iterable[Tuple[str, TJson]]


def read_json(fpath: Text) -> TJson:
    """Read a json file and return the content."""
    logger.debug(f"Reading: json file: {fpath}")
    with open(fpath, "r", encoding="utf8") as file:
        fcontent = json.load(file)
    logging.debug(f"Loaded: {fpath}")
    return fcontent


def read_pickle(fpath: Text) -> Any:
    """Read a pcikle file and return the content."""
    logger.debug(f"Reading: pickle file: {fpath}")
    with open(fpath, "rb",) as file:
        fcontent = pickle.load(file)
    logging.debug(f"Loaded: {fpath}")
    return fcontent


def write_json(fpath: Text, fcontent: TJson, override: bool = False, indent: int = 0, **kwargs) -> None:
    """Write a json file."""
    logger.debug(f"Writing: json file: {fpath}")
    if path.exists(fpath):
        if not override:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), fpath)
        else:
            logger.debug(f"Overriding: {fpath}")
    path_dir = path.split(fpath)[0]
    if path_dir:
        os.makedirs(path_dir, exist_ok=True)
    with open(fpath, "w", encoding="utf8") as file:
        fcontent = json.dump(fcontent, file, ensure_ascii=False, indent=indent)
    logging.debug(f"Written: {fpath}")


def write_pickle(fpath: Text, fcontent: TJson, override: bool = False, **kwargs) -> None:
    """Write a pickle file."""
    logger.debug(f"Writing: pickle file: {fpath}")
    if path.exists(fpath):
        if not override:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), fpath)
        else:
            logger.debug(f"Overriding: {fpath}")
    path_dir = path.split(fpath)[0]
    if path_dir:
        os.makedirs(path_dir, exist_ok=True)
    with open(fpath, "wb") as file:
        fcontent = pickle.dump(fcontent, file)
    logging.debug(f"Written: {fpath}")


class DataLoader(abc.ABC):
    """Data loaders base class."""

    def __init__(
        self,
        datapath: Union[Text, Iterable[Text]],
        fileformat: Text,
        dataformat: Text = "default",
        sort_filename: bool = True,
        skip_file_cb: Optional[Callable[[Text], bool]] = None,
    ):
        datapath = [datapath] if isinstance(datapath, Text) else list(datapath)
        self._datapath = datapath

        valid_fileformat = ["json", "pickle"]
        fileformat = fileformat.lower()
        if fileformat not in valid_fileformat:
            raise ValueError(f"invalid `fileformat`: '{fileformat}'; must be one of {valid_fileformat}")
        self._fileformat = fileformat

        self._dataformat = dataformat
        self.sort_filename: bool = sort_filename
        self.skip_file_cb: Callable = skip_file_cb

        self._num_files = len(self.filepaths())
        self._paths_it = iter(self.filepaths())
        self._i = 0
        if fileformat == "json":
            self._reader = read_json
        elif fileformat == "pickle":
            self._reader = read_pickle

    @property
    def datapath(self) -> List[Text]:
        """List of path passed at initialization."""
        return self._datapath

    @property
    def fileformat(self) -> Text:
        """File format."""
        return self._fileformat

    @property
    def dataformat(self) -> Text:
        """Data format. Does not influence the instance behaviour."""
        return self._dataformat

    @property
    def num_files(self) -> int:
        """Number of files in the dataset."""
        return self._num_files

    @abc.abstractmethod
    def filepaths(self) -> List[Text]:
        """The list of file paths in the dataset, either sorted or randomized."""
        ...

    def __iter__(self) -> DataIterable:
        self._paths_it = iter(self.filepaths())
        self._i = 0
        return self

    def __next__(self) -> Any:
        fpath = next(self._paths_it)
        self._i += 1
        logger.info(f"[{self._i} / {self.num_files}] {fpath}")
        if self.skip_file_cb is not None and self.skip_file_cb(fpath):
            logger.info(f"Skipped!")
            return next(self)
        return fpath, self._reader(fpath)


class FileDataLoader(DataLoader):
    """Data loader for single/multiple files."""

    def filepaths(self) -> List[Text]:
        """The list of file paths in the dataset, either sorted or randomized."""
        if self.sort_filename:
            return list(sorted(self._datapath))
        else:
            paths = copy.copy(self._datapath)
            rd.shuffle(paths)
            return paths


class DirDataLoader(DataLoader):
    """Data loader for single/multiple directory."""

    def __init__(
        self,
        datapath: Union[Text, Iterable[Text]],
        fileformat: Text,
        dataformat: Text = "default",
        sort_filename: bool = True,
    ):
        self._paths = None
        super().__init__(datapath, fileformat, dataformat, sort_filename)

    def filepaths(self) -> List[Text]:
        """The list of file paths in the dataset, either sorted or randomized."""
        if self._paths is None:
            self._paths = []
            for dirpath in self.datapath:
                self._paths.extend(self.discover_files(dirpath, self.fileformat))
        if self.sort_filename:
            return list(sorted(self._paths))
        else:
            paths = copy.copy(self._paths)
            rd.shuffle(paths)
            return paths

    @staticmethod
    def discover_files(dirpath: Text, extension: Text) -> List[Text]:
        """Find the files with `extension` in the directory `dir_path` and its subdirectories."""
        paths = []
        extension = extension.strip(".").lower()
        for subdirpath, _, files in os.walk(dirpath):
            for filename in files:
                if path.splitext(filename)[1].strip(".").lower() == extension:
                    paths.append(path.join(subdirpath, filename))
        return paths


class DataDumper:
    """Data dumper class, and path modification features."""

    def __init__(
        self,
        fileformat: Text,
        path_modifier: Callable[[Text], Text] = None,
        override: bool = False,
        json_indent: int = 0,
    ):
        valid_fileformat = ["json", "pickle"]
        fileformat = fileformat.lower()
        if fileformat not in valid_fileformat:
            raise ValueError(f"invalid `fileformat`: '{fileformat}'; must be one of {valid_fileformat}")
        self._fileformat = fileformat

        self.path_modifier = path_modifier or self.noop_path_mod
        self.override = override
        self.json_indent = json_indent
        if self.fileformat == "json":
            self._writer = write_json
        elif self.fileformat == "pickle":
            self._writer = write_pickle

    @property
    def fileformat(self) -> Text:
        """File format."""
        return self._fileformat

    def save(self, iterable: Iterable[Tuple[Text, Any]]) -> None:
        """Save the elements in iterable."""
        for fpath, fcontent in iterable:
            self._writer(self.path_modifier(fpath), fcontent, self.override, indent=self.json_indent)

    @staticmethod
    def dir_replacer(from_dir: Text, to_dir: Text) -> Callable[[Text], Text]:
        """Return a path modifier function replacing the last occurence of `from_dir` to `to_dir` in the path."""

        def path_modifier(fpath: Text) -> Text:
            dirs, filename = path.split(fpath)
            sdirs = dirs.split("/")
            try:
                i = len(sdirs) - 1 - sdirs[::-1].index(from_dir)
            except ValueError:
                raise ValueError(f"Replacing {from_dir} to {to_dir}: {from_dir} not part of the path {fpath}")
            else:
                new_spath = sdirs[:i] + [to_dir] + sdirs[i + 1 :] + [filename]
                return path.join(*new_spath)

        return path_modifier

    @staticmethod
    def path_replacer(new_path: Text) -> Callable[[Text], Text]:
        """Return a path modifier function replacing any path with `new_path`."""
        return lambda _: new_path

    @staticmethod
    def path_in_dir(new_dir: Text) -> Callable[[Text], Text]:
        """Return a path modifier function joining the path to `new_dir`."""
        return lambda fpath: path.join(new_dir, fpath)

    @staticmethod
    def file_in_dir(new_dir: Text) -> Callable[[Text], Text]:
        """Return a path modifier function joining the path filename to `new_dir`."""
        return lambda fpath: path.join(new_dir, path.split(fpath)[1])

    def make_skip_cb(self):
        """Return a function taking an input path as argument and check if the transformed path exists."""
        if self.override:
            return lambda _: False
        return lambda fpath: path.exists(self.path_modifier(fpath))

    @staticmethod
    def noop_path_mod(fpath):
        """Returns its argument."""
        return fpath
