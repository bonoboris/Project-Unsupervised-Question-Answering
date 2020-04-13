"""
CLI helper functions and decorators.
"""
import functools
from os import path
from typing import Callable, List

import click

from uqa.dataset import DataDumper, DirDataLoader, FileDataLoader


# def _string_pair_cb(ctx: click.Context, _: click.Parameter, value: str) -> List[Tuple[str, str]]:
#     ret = list()
#     if value is None:
#         return ret
#     try:
#         pairs = value.split(";")
#         for pair in pairs:
#             ret.append(pair.split(":"))
#     except ValueError:
#         raise click.BadParameter("Invalid format: expected 'from_dir_1:to_dir_1;from_dir_2:to_dir_2'")
#     else:
#         return ret


def _validate_params(use_dir: bool, src: List[str]) -> bool:
    if use_dir:
        if len(src) != 1:
            raise click.BadParameter("Single SRC allowed when using -d / --dir flag.", param_hint="[SRC]...")
        if not path.isdir(src[0]):
            raise click.BadParameter(
                f"'{src[0]}' is file; SRC must be a directory when using -d / --dir flag.", param_hint="[SRC]..."
            )
    else:
        for src_path in src:
            if not path.isfile(src_path):
                raise click.BadParameter(
                    f"'{src_path}' is a directory; use -d /--dir flag to pass a directory path.", param_hint="[SRC]..."
                )


def _read_params(func: Callable, default_dataformat_only: bool = False) -> Callable:
    decorated_func = func
    if not default_dataformat_only:
        decorated_func = click.option(
            "-df",
            "--dataformat",
            type=click.Choice(["default", "fsquad"], case_sensitive=False),
            default="default",
            help="Data structure",
        )(decorated_func)
    decorated_func = click.option(
        "-if",
        "--input-format",
        type=click.Choice(["json", "pickle"], case_sensitive=False),
        default="json",
        show_default=True,
        help="Input file(s) format.",
    )(decorated_func)
    decorated_func = click.option(
        "-d",
        "--dir",
        "use_dir",
        is_flag=True,
        is_eager=True,
        expose_value=True,
        help="Indicate `SRC` value is a directory.",
    )(decorated_func)
    decorated_func = click.argument("src", nargs=-1, type=click.Path(exists=True))(decorated_func)

    return decorated_func


def click_read_data(func: Callable) -> Callable:
    """Add parameters for commands which read data files or directory,
    and transform those parameters in a DataLoader instance passed the decorated function
    as the keyword argument `dataloader`."""

    @functools.wraps(func)
    def wrapper(dataformat, input_format, use_dir, src, **kwargs):
        _validate_params(use_dir, src)
        if use_dir:
            dataloader = DirDataLoader(src, input_format, dataformat)
        else:
            dataloader = FileDataLoader(src, input_format, dataformat)
        return func(dataloader=dataloader, **kwargs)

    decorated_func = _read_params(wrapper)
    doc_str = (
        "\nRead and process SRC data. SRC can be one or more path(s) to files.\n"
        "With -d / --dir flag, SRC must be a single path to a directory. "
        "File in SRC and its sub-directories are discoverd and "
        "processed if they have the right extension ('*.json' or '*.pickle')\n"
    )
    if not decorated_func.__doc__.endswith("\n"):
        decorated_func.__doc__ += "\n"
    decorated_func.__doc__ += doc_str

    return decorated_func


def _write_params(func: Callable) -> Callable:
    decorated_func = func
    decorated_func = click.option(
        "--json-indent", type=click.INT, default=0, help="Json indentation number of space (0 for compact)"
    )(decorated_func)
    decorated_func = click.option(
        "-of",
        "--output-format",
        type=click.Choice(["json", "pickle"], case_sensitive=False),
        default="json",
        show_default=True,
        help="Output file format.",
    )(decorated_func)
    decorated_func = click.option("-O", "--override", is_flag=True, help="Override existing output files")(
        decorated_func
    )
    decorated_func = click.argument("dst", nargs=1, type=click.Path())(decorated_func)

    return decorated_func


def click_read_write_data(func: Callable) -> Callable:
    """Add parameters for commands which read, process and write data files or directory,
    and transform those parameters in a DataLoader instance and a DataDumper instance passed to the decorated function
    as the keyword argument `datloader` and `datadumper`."""

    @functools.wraps(func)
    def wrapper(dataformat, input_format, use_dir, src, output_format, json_indent, override, dst, **kwargs):
        _validate_params(use_dir, src)
        if use_dir:
            dataloader = DirDataLoader(src, input_format, dataformat)
            path_mod = DataDumper.dir_replacer(src[0].strip("/"), dst)
        else:
            dataloader = FileDataLoader(src, input_format, dataformat)
            if len(src) == 1:
                path_mod = DataDumper.path_replacer(dst)
            else:
                path_mod = DataDumper.file_in_dir(dst)
        datadumper = DataDumper(output_format, path_mod, override=override, json_indent=json_indent)
        dataloader.skip_file_cb = datadumper.make_skip_cb()
        return func(dataloader=dataloader, datadumper=datadumper, **kwargs)

    decorated_func = _read_params(_write_params(wrapper))

    doc_str = (
        "\nRead and process SRC data. SRC can be one or more path(s) to files. "
        "With -d / --dir flag, SRC must be a single path to a directory. "
        "File in SRC and its sub-directories are discoverd and "
        "processed if they have the right extension ('*.json' or '*.pickle').\n"
        "\nWrite processed SRC data in path DST. "
        "If SRC is a single file path, write the processed data the file DST. "
        "If SRC is a single directory path, write the processed data in the directory DST "
        "retaining SRC internal hierarchy. "
        "If SRC is a list of files path: write the processed data in the directory DST with the same filename.\n"
    )
    if not decorated_func.__doc__.endswith("\n"):
        decorated_func.__doc__ += "\n"
    decorated_func.__doc__ += doc_str

    return decorated_func


def click_split_params(func: Callable) -> Callable:
    """Add parameters for commands which read, process and write data files or directory,
    and transform those parameters in a DataLoader instance and a DataDumper instance passed to the decorated function
    as the keyword argument `datloader` and `datadumper`."""

    @functools.wraps(func)
    def wrapper(dataformat, input_format, use_dir, src, output_format, override, dst, **kwargs):
        _validate_params(use_dir, src)
        if use_dir:
            dataloader = DirDataLoader(src, input_format, dataformat)
        else:
            dataloader = FileDataLoader(src, input_format, dataformat)

        datadumper = DataDumper(output_format, override=override)
        return func(dataloader=dataloader, datadumper=datadumper, dst=dst, **kwargs)

    decorated_func = click.argument("num", type=int, required=True)(_read_params(_write_params(wrapper)))

    doc_str = (
        "\nRead and process SRC data. SRC can be one or more path(s) to files. "
        "With -d / --dir flag, SRC must be a single path to a directory. "
        "File in SRC and its sub-directories are discoverd and "
        "processed if they have the right extension ('*.json' or '*.pickle').\n"
        "\nSplit / combine data in file with NUM articles per file and save them in path "
        "generated with template DST. If NUM is <= 0 a single file is created with all the articles,"
        "else DST must contains '{num}' placeholder (ex: DST='foo/bar_{num}.json') in which case"
        "files path are generated by replacing the placeholder by numbers starting from 1 and incrementing.\n"
    )
    if not decorated_func.__doc__.endswith("\n"):
        decorated_func.__doc__ += "\n"
    decorated_func.__doc__ += doc_str

    return decorated_func
