"""
CLI commands definition with click.
"""
from typing import List

import click

from uqa import (
    logging_utils,
    dataset,
    fsquad_utils,
    cli_helpers,
    stats as stats_,
    clean as clean_,
    split as split_,
    ner as ner_,
    constituency as constituency_,
    qa_gen,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("--quiet", "--silent", is_flag=True, help="Disable console logging")
@click.option("--no-log", is_flag=True, help="Disable file logging")
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    show_default=True,
    help="Console logging level",
)
@click.option(
    "-lf",
    "--log-file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    default="uqa.log",
    show_default=True,
    help="Log file path",
)
@click.option(
    "-lv",
    "--log-verbosity",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="debug",
    show_default=True,
    help="File logging level",
)
def main(quiet, no_log, verbosity, log_file, log_verbosity):
    """Project UQA command line tool."""
    logging_utils.init_root_logger(quiet, no_log, log_file, verbosity.upper(), log_verbosity.upper())


@main.command()
@click.option("--detailed", is_flag=True, help="Log per-file stats")
@cli_helpers.click_read_data
def stats(dataloader: dataset.DataLoader, detailed: bool):
    """Count articles and contexts."""
    stats_.stats_dl(dataloader, detailed)


@main.command()
@click.option(
    "-a",
    "--action",
    type=click.Choice(["clean", "filter"], case_sensitive=False),
    multiple=True,
    required=True,
    default=["clean", "filter"],
    show_default=True,
    help="Action to be performed (allows multiple options).",
)
@click.option(
    "--filter-min-alpha",
    type=click.INT,
    default=10,
    show_default=True,
    help="Minimum number of letters for a context to be valid",
)
@click.option("--detailed", is_flag=True, help="Log per-file information.")
@cli_helpers.click_read_write_data
def clean(
    dataloader: dataset.DataLoader,
    datadumper: dataset.DataDumper,
    action: List[str],
    filter_min_alpha: int,
    detailed: bool,
):
    """Clean data."""
    data_it = dataloader
    if dataloader.dataformat == "fsquad":
        data_it = fsquad_utils.fsquad_to_default_dl(data_it)
    if "clean" in action:
        data_it = clean_.clean_dl(data_it)
    if "filter" in action:
        data_it = clean_.filter_contexts_dl(data_it, filter_min_alpha, detailed)
    datadumper.save(data_it)


@main.command()
@cli_helpers.click_split_params
def split(dataloader: dataset.DataLoader, datadumper: dataset.DataDumper, dst: str, num: int):
    """Split / combine data in file(s)."""
    if num < 1:
        data_it = split_.unite_dl(dataloader, dst)
    else:
        data_it = split_.split_dl(dataloader, dst, num)
    datadumper.save(data_it)


@main.command()
@cli_helpers.click_read_write_data
def ner(dataloader: dataset.DataLoader, datadumper: dataset.DataDumper):
    """Named-entity recognition."""
    datadumper.save(ner_.ner_dl(dataloader))


@main.command()
@cli_helpers.click_read_write_data
def constituency(dataloader: dataset.DataLoader, datadumper: dataset.DataDumper):
    """Constituency parsing."""
    datadumper.save(constituency_.constituency_dl(dataloader))


@main.command()
@cli_helpers.click_read_write_data
def qas(dataloader: dataset.DataLoader, datadumper: dataset.DataDumper):
    """Natural question / answer genration."""
    datadumper.save(fsquad_utils.default_to_squad_dl(qa_gen.generate_qas_dl(dataloader)))
