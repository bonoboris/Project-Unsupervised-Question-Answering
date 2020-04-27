"""
Classes to load and save data.

:class:`DataLoader` and :class:`DataDumper` are designed to work with/as :obj:`DataIterable` types,
while encapsulating the different dataset structure and format specificities.

Examples
--------
Read a single file pickle dataset, process it and save a single json file::

    >>> # Load a single pickle file
    >>> dataloader = FileDataLoader("data.pickle", fileformat="pickle")
    >>> # Processing returns an `DataIterble` object with a single element
    >>> data_it = some_processing_step(dataloader)
    >>> # Get a path_modifier which simply returns a new path
    >>> path_modifier = DataDumper.path_replacer("foo/data.json")
    >>> DataDumper(
    ...     fileformat="json",              # save in JSON fileformat
    ...     path_modifier=path_modifier,    # use new path
    ...     json_indent=2,                  # set json indentation level to 2
    ... ).save(data_it)                     # Save the processed data
    >>> # File 'foo/data.json' is created and contains the processed data
    >>> # if 'foo' directory doesn't exist, it's created.

Read a multiple-file json dataset, process it and write result in a new directory.::

    >>> # Load multiple json file
    >>> dataloader = FileDataLoader(["data_train.json", "bar/data_eval.json", "bar/data_test.json"])
    >>> # Processing returns an `DataIterble` object with a 3 elements.
    >>> data_it = some_processing_step(dataloader)
    >>> # Get the path modifier
    >>> path_modifier = DataDumper.file_in_dir("foo")
    >>> DataDumper(
    ...     fileformat="json",              # save in JSON fileformat
    ...     path_modifier=path_modifier,    # use new path
    ...     json_indent=2,                  # set json indentation level to 2
    ... ).save(data_it)                     # Save the processed data
    >>> # Files 'foo/data_train.json', 'foo/data_eval.json', 'foo/data_test.json'] are created
    >>> # if 'foo' directory doesn't exist, it's created.

Read a directory json dataset, process it and write result in a new directory.::

    >>> # Load all json file in directory "bar" and its sub-directories
    >>> dataloader = DirDataLoader("bar")
    >>> # Processing returns an `DataIterble` object.
    >>> data_it = some_processing_step(dataloader)
    >>> # Get the directory modifier function
    >>> path_modifier = DataDumper.dir_replacer("bar", "baz/foo")
    >>> DataDumper(
    ...     fileformat="json",              # save in JSON fileformat
    ...     path_modifier=path_modifier,    # use new path
    ...     json_indent=2,                  # set json indentation level to 2
    ... ).save(data_it)                     # Save the processed data
    >>> # For each file with path "bar/{subpath}" a file "foo/baz/{subpath}" is created.
    >>> # All non-exsting directories and sub-directories are also created
"""

import abc
import copy
import errno
import logging
import os
import pickle
import random as rd
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import ujson as json
except ModuleNotFoundError:
    import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

#: | Json-like container, representing data usually in `default` format.
#: | See -- :doc:`/data_formats`
TJson = Union[Dict, List]

#: Dataset Iterable, each element is a pair ``(path, fcontent)``
#:
#: * **path** (`str`) -- File's path
#: * **fcontent** (:obj:`TJson`) -- File's content
DataIterable = Iterable[Tuple[str, TJson]]


def read_json(fpath: str) -> TJson:
    """Read a json file and return its content.

    Parameters
    ----------
    fpath: str
        Relative or absolute path of the JSON file to read

    Returns
    -------
    :obj:`TJson`
        The JSON file's content
    """
    logger.debug(f"Reading: json file: {fpath}")
    with open(fpath, "r", encoding="utf8") as file:
        fcontent = json.load(file)
    logging.debug(f"Loaded: {fpath}")
    return fcontent


def read_pickle(fpath: str) -> Any:
    """Unpickle a file and return its content.

    Parameters
    ----------
    fpath: str
        Relative or absolute path of the pickle file to read

    Returns
    -------
    Any
        The unpickled object
    """
    logger.debug(f"Reading: pickle file: {fpath}")
    with open(fpath, "rb",) as file:
        fcontent = pickle.load(file)
    logging.debug(f"Loaded: {fpath}")
    return fcontent


def write_json(fpath: str, fcontent: TJson, override: bool = False, indent: int = 0, **kwargs) -> None:
    """Write `fcontent` json-like structure at path `fpath`.

    Parameters
    ----------
    fpath: str
        Relative or absolute path of the file to write.
    fcontent: :obj:`TJson`
        Json-like content to write.
    override: bool, default=False
        If a file with path `fpath` already exists and overriden is ``True`` the file is overriden,
        else if override is ``False`` a :exc:`FileExistsError` exception is raised.
    indent: int, default=0
        JSON indentation number of whitespace to use, use 0 for compact JSON

    Keyword Args
    ------------
    kwargs
        Ignored.
    """
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


def write_pickle(fpath: str, fcontent: Any, override: bool = False, **kwargs) -> None:
    """Picke and write `fcontent` object at path `fpath`.

    Parameters
    ----------
    fpath: str
        Relative or absolute path of the file to write.
    fcontent: :obj:`TJson`
        Picklable object to write.
    override: bool, default=False
        If a file with path `fpath` already exists and overriden is ``True`` the file is overriden,
        else if override is ``False`` a :exc:`FileExistsError` exception is raised.

    Keyword Args
    ------------
    kwargs
        Ignored.
    """
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
    """Data loaders base class.

    | Derived classes only need to implement :meth:`filepaths` abstract methods.
    | Manage dataset reading for single or multiple files dataset and allow to specify a directory as a dataset.
    | Iterating over an `DataLoader` instance will yield each file path and content.

    Attributes
    ---------
    sort_filename: bool, default=True
        If ``True`` file path and content pairs are sorted in lexicographic order relative to the path.
        else files order is randomized
    skip_file_cb: Callable[[str], bool], default=None
        If provided, each file's path in the dataset is passed to this callback, if the callback returns ``True``
        the file is skipped

    See Also
    --------
    :obj:`DataIterable`
    """

    def __init__(
        self,
        datapath: Union[str, Iterable[str]],
        fileformat: str,
        dataformat: str = "default",
        sort_filename: bool = True,
        skip_file_cb: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """
        Parameters
        ----------
        datapath: str or iterable of str
            | Path(s) to a file or directory.
            | A path to a directory will be recursively explored and files with the correct extension
              will be considered part of the dataset.
        fileformat: str, ["json", "pickle"], default="json"
            | A supported fileformat.
            | With ``"json"`` fileformat only ``*.json`` will be matched when exploring a folder.
            | With ``"pickle"`` fileformat only ``*.pickle`` will be matched when exploring a folder.
        dataformat: str, ["default", "fquad"], default="default"
            | One of the supported data format.
            | The given value is only stored and does not affect the instance behaviour.
        sort_filename: bool, default=True
            If ``True`` file path and content pairs are sorted in lexicographic order relative to the path.
            else files order is randomized
        skip_file_cb: Callable[[str], bool], default=None
            If provided, each file's path in the dataset is passed to this callback, if the callback returns ``True``
            the file is skipped
        """
        datapath = [datapath] if isinstance(datapath, str) else list(datapath)
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
    def datapath(self) -> List[str]:
        """list of str: List of path passed at initialization."""
        return self._datapath

    @property
    def fileformat(self) -> str:
        """str: File format, one of (`json`, `pickle`)."""
        return self._fileformat

    @property
    def dataformat(self) -> str:
        """str: Data format, one of (`default`, `fquad`)."""
        return self._dataformat

    @property
    def num_files(self) -> int:
        """int: Number of files in the dataset."""
        return self._num_files

    @abc.abstractmethod
    def filepaths(self) -> List[str]:
        """Return the list of file paths in the dataset, order depends on :attr:`sort_filename` value."""
        ...

    def __iter__(self) -> DataIterable:
        """Use :meth:`filepaths` method result to iterate over the dataset.

        Returns
        -------
        :obj:`.DataIterable`:
            `self`
        """
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

    def filepaths(self) -> List[str]:
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
        datapath: Union[str, Iterable[str]],
        fileformat: str,
        dataformat: str = "default",
        sort_filename: bool = True,
    ):
        self._paths = None
        super().__init__(datapath, fileformat, dataformat, sort_filename)

    def filepaths(self) -> List[str]:
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
    def discover_files(dirpath: str, extension: str) -> List[str]:
        """Recursively find the files with `extension` in the directory `dir_path` and its subdirectories.

        Parameters
        ----------
        dirpath: str
            The path of the folder to explore
        extension: str
            The target extension

        Returns
        -------
        list of str
            The list of files in `dirpath` directory and its sub-directories with extension `extension`.
        """
        paths = []
        extension = extension.strip(".").lower()
        for subdirpath, _, files in os.walk(dirpath):
            for filename in files:
                if path.splitext(filename)[1].strip(".").lower() == extension:
                    paths.append(path.join(subdirpath, filename))
        return paths


class DataDumper:
    """Data dumping class.

    | Implement fetures for saving a :obj:`DataIterable` in JSON or Pickle format as well
      as path modifier factories.

    Attributes
    ----------
    path_modifier: Callable[[str], str], default=no-op
        A callback to transform path.
    override: bool, default=False
        Flag setting existing file overriding behaviour
    json_indent: int, default=0
        JSON indentation number of whitespace, ignored if using Pickle fileformat
    """

    def __init__(
        self, fileformat: str, path_modifier: Callable[[str], str] = None, override: bool = False, json_indent: int = 0,
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
    def fileformat(self) -> str:
        """str: File format, one of (`json`, `pickle`)."""
        return self._fileformat

    def save(self, data_it: DataIterable) -> None:
        """Apply :attr:`path_modifier` and dump the elements in `data_it`.

        Parameters
        ----------
        data_it: DataIterable
            An iterator of the data to save.
        """
        for fpath, fcontent in data_it:
            self._writer(self.path_modifier(fpath), fcontent, self.override, indent=self.json_indent)

    def make_skip_cb(self):
        """Return a function taking an input path as argument and check if the modified path exists.

        The modified path of a given path ``fpath`` is the result of applying :attr:`path_modifier` to ``fpath``
        """
        if self.override:
            return lambda _: False
        return lambda fpath: path.exists(self.path_modifier(fpath))

    @staticmethod
    def noop_path_mod(fpath):
        """A function returning its argument without modification."""
        return fpath

    @staticmethod
    def dir_replacer(from_dir: str, to_dir: str) -> Callable[[str], str]:
        """Function factory returning a path modifier function replacing the last occurence
        of `from_dir` to `to_dir` in the path.

        Parameters
        ----------
        from_dir: str
            A directory name (doesn't allow multiple directory)
        to_dir: str
            A string to substitute `from_dir` with (accepts mutliple dir, i.e. "foo/bar")

        Returns
        -------
        Callable[[str], str]
            A function taking a '/'-path string as an argument and replacing the last occurence
            of `from_dir` with `to_dir`.
        """

        def path_modifier(fpath: str) -> str:
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
    def path_replacer(new_path: str) -> Callable[[str], str]:
        """Function factory returning a path modifier function replacing the whole path with `new_path`.

        | Returns ``lambda _: new_path``
        | Used for single file dataset.

        Parameters
        ----------
        new_path: str
            The `new_path` to return

        Returns
        -------
        Callable[[str], str]
            A function taking a single argument and returning `new_path`.
        """
        return lambda _: new_path

    @staticmethod
    def path_in_dir(new_dir: str) -> Callable[[str], str]:
        """Function factory returning a path modifier function joining `new_dir` and its path argument.

        | Returns ``lamda fpath: os.path.join(new_dir, fpath)``

        Parameters
        ----------
        new_dir: str
            A directory name (accepts multiple directory, i.e. "foo/bar")

        Returns
        -------
        Callable[[str], str]
            A function taking a '/'-path string argument `fpath` and returning the joined path of
            `fpath` in directory `new_dir`.
        """
        return lambda fpath: path.join(new_dir, fpath)

    @staticmethod
    def file_in_dir(new_dir: str) -> Callable[[str], str]:
        """Function factory returning a path modifier function joining `new_dir` and its path argument filename.

        Parameters
        ----------
        new_dir: str
            A directory name (accepts multiple directory, i.e. "foo/bar")

        Returns
        -------
        Callable[[str], str]
            A function taking a '/'-path string argument `fpath` and returning the joined path of
            `fpath` **filename** in directory `new_dir`.
        """
        return lambda fpath: path.join(new_dir, path.split(fpath)[1])
