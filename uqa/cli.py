"""CLI entry point and command definitions with click.

The CLI is designed to wrap the maximum number of features implemented in the package.
To use the CLI the `uqa` package must be installed (see `setup.py`).

Notes
-----
Docstring of functions implementing CLI commands are used as help text in the CLI.

For all commands reading and/or writing data decorators defined in `cli_helpers.py` are used.
Those decorators achieve goals:
- Adding commands parameters
- Extending function docstring with usage documentation for CLI help text
- At runtime, transforming parameters in :class:`dataset.DataLoader` and / or :class:`dataset.DataDumper` instances
"""

from typing import List

import click

from uqa import (
    logging_utils,
    dataset,
    fquad_utils,
    cli_helpers,
    stats as stats_,
    clean as clean_,
    split as split_,
    ner as ner_,
    constituency as constituency_,
    show as show_,
    validate as validate_,
    download as download_,
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
    if dataloader.dataformat == "fquad":
        data_it = fquad_utils.fquad_to_default_dl(data_it)
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
    data_it = dataloader
    if dataloader.dataformat == "fquad":
        data_it = fquad_utils.fquad_to_default_dl(dataloader)
    datadumper.save(ner_.ner_dl(data_it))


@main.command()
@click.option("--detailed", is_flag=True, help="Log article processing progression")
@cli_helpers.click_read_write_data
def constituency(dataloader: dataset.DataLoader, datadumper: dataset.DataDumper, detailed: bool):
    """Constituency parsing."""
    data_it = dataloader
    if dataloader.dataformat == "fquad":
        data_it = fquad_utils.fquad_to_default_dl(dataloader)
    datadumper.save(constituency_.constituency_dl(data_it, detailed=detailed))


@main.command()
@cli_helpers.click_read_write_data
def qas(dataloader: dataset.DataLoader, datadumper: dataset.DataDumper):
    """Natural question / answer genration."""
    datadumper.save(fquad_utils.default_to_fquad_dl(qa_gen.generate_qas_dl(dataloader)))


@main.command()
@click.option("-a", "--all", "show_all", is_flag=True, help="Show all context")
@click.option("--depth", type=click.INT, default=-1, help="Maximum depth for constituents")
@click.option("--no-ner", is_flag=True)
@click.option("--no-const", is_flag=True)
@click.option("--show-no-label", is_flag=True)
@click.option("--rule", type=click.Choice(["", "rule1", "rule1_ext"]), default="")
@cli_helpers.click_read_data
def show(
    dataloader: dataset.DataLoader,
    depth: int,
    show_all: bool,
    no_ner: bool,
    no_const: bool,
    show_no_label: bool,
    rule: str,
):
    """Show colorized context and informations."""
    data_it = dataloader
    if dataloader.dataformat == "fquad":
        data_it = fquad_utils.fquad_to_default_dl(data_it, include_qas=True)
    if rule:
        show_.show_rule_dl(data_it, rule, show_all)
    else:
        show_.show_dl(data_it, depth, show_all=show_all, no_ner=no_ner, no_const=no_const, show_no_label=show_no_label)


@main.command()
@cli_helpers.click_read_data
def validate(dataloader: dataset.DataLoader):
    """Validate a dataset in `FQuAD` format."""
    if dataloader.dataformat != "fquad":
        raise click.BadParameter(f"Unsupported `dataformat`: '{dataloader.dataformat}'")
    validate_.validate_dl(dataloader)


@main.command()
@click.argument(
    "model", type=click.Choice(["all", "spacy", "nltk"], case_sensitive=False), default="all",
)
@click.option("-n", "--name", type=click.STRING, multiple=True, help="Name of a specific model to install.")
def download(model: str, name: List[str]):
    """Download models and ressources.

    `MODEL` argument refers to the model type. Choose `all` to download all default required models.
    """
    if model == "all" and name:
        click.BadParameter("Cannot specify model name with `model` value `all`")
    if model == "all":
        download_.download_spacy_model()
        download_.download_benepar_model()
    else:
        dl_func = download_.download_spacy_model if model == "spacy" else download_.download_benepar_model
        if name:
            for model_name in name:
                dl_func(model_name)
        else:
            dl_func()
