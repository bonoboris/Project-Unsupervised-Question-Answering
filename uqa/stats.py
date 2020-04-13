"""Compute and log simple dataset stats."""

from collections import Counter
import logging
from typing import Counter as TCounter, Mapping

from uqa.dataset import DataLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def mapping_str(mapping: Mapping, item_str_template: str = "{k}: {v}", item_sep=" | ") -> str:
    """Returns a representation of a mapping"""
    sub_strs = [item_str_template.format(k=k, v=v) for k, v in mapping.items()]
    return item_sep.join(sub_strs)


def stats(fcontent, dataformat="default") -> TCounter[str]:
    """Return the simple stats over a data collection."""
    counts = Counter()
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


def stats_dl(dataloader: DataLoader, detailed=True) -> TCounter[str]:
    """Return the simple stats over a data collection."""
    counts: TCounter[str] = Counter()
    for _, fcontent in dataloader:
        fcounts = stats(fcontent, dataloader.dataformat)
        counts.update(fcounts)
        if detailed:
            logger.info(mapping_str(fcounts))
    logger.info(f"TOTAL: {mapping_str(counts)}")
    return counts
