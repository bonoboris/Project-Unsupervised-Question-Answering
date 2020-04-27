"""Compute and log simple dataset stats."""

import collections
import logging
from typing import Counter as TCounter, Mapping

from uqa import dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def mapping_str(mapping: Mapping, item_str_template: str = "{k}: {v}", item_sep=" | ") -> str:
    """Returns a string representation of a mapping.

    Parameters
    ----------
    mapping: Mapping
        A mapping
    item_str_template: str
        A template string with `{k}` and `{v}` placeholders for the entries key and value respectively
    item_sep: str
        The separator to use between mapping entries string.

    Returns
    -------
    str
        The mapping representation as a string
    """
    sub_strs = [item_str_template.format(k=k, v=v) for k, v in mapping.items()]
    return item_sep.join(sub_strs)


def stats(fcontent: dataset.TJson, dataformat: str = "default") -> TCounter[str]:
    """Return simple stats over `fcontent`.

    | Count the numbers of `articles` and `contexts`.
    | For `fquad` dataformat also count the number of `questions`

    Parameters
    ----------
    fcontent: :obj:`dataset.TJson`
        The data container
    dataformat: str, default="default"
        The data format

    Returns
    -------
    collections.Counter
        A Counter instance with entries `articles` and `contexts` and optionaly `questions`.
    """
    counts = collections.Counter()
    if dataformat == "default":
        counts["articles"] = len(fcontent)
        counts["contexts"] = sum((len(art["contexts"]) for art in fcontent))
    elif dataformat == "fquad":
        data = fcontent["data"]
        counts["articles"] = len(data)
        for art in data:
            counts["contexts"] += len(art["paragraphs"])
            counts["questions"] += sum((len(para["qas"]) for para in art["paragraphs"]))
    return counts


def stats_dl(dataloader: dataset.DataLoader, detailed: bool = True) -> TCounter[str]:
    """Return simple stats over the :obj:`dataset.DataLoader` instance `dataloader`.

    | For `default` format dataset count the number `articles` and `contexts`.
    | For `fquad` format dataset also count the number of `questions`

    Parameters
    ----------
    dataloader: :obj:`dataset.DataLoader`
        A :obj:`dataset.DataLoader` instance.
    detailed: bool, default=True
        If ``True`` logs per file stats

    Returns
    -------
    collections.Counter
        A Counter instance with entries `articles` and `contexts` and optionaly `questions`.
    """
    counts: TCounter[str] = collections.Counter()
    for _, fcontent in dataloader:
        fcounts = stats(fcontent, dataloader.dataformat)
        counts.update(fcounts)
        if detailed:
            logger.info(mapping_str(fcounts))
    logger.info(f"TOTAL: {mapping_str(counts)}")
    return counts
