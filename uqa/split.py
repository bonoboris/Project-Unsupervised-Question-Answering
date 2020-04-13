"""
Split and structure a dataset.
"""

import logging
import math

from uqa import dataset, stats

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def unite_dl(data_it: dataset.DataIterable, output_fpath: str) -> dataset.DataIterable:
    """Concatenate the data in the data iterable into a single list and yield (`output_path`, concatenated data)."""
    new_fcontent = list()
    for _, data in data_it:
        new_fcontent.extend(data)
    yield output_fpath, new_fcontent


def split_dl(dataloader: dataset.DataLoader, fpath_template: str, num_artcles_per_file: int) -> dataset.DataIterable:
    """Split the dataloader data into `num_articles_per_file` articles splits and generate a new path with the template

    Args
    ----
        dataloader: dataset.Dataloader
            A dataloader instance
        fpath_template: str
            A bracket style path template string with a '{num}' placeholder.
            '{num}' is formated with the split number automaticlly padded with zeros.
        num_artcles_per_file: int
            The new number of article per file
    Yield
    -----
        (fpath, fcontent): (str, Json-like)
            the new path along, the new data

    """
    total_num_articles = stats.stats_dl(dataloader, detailed=False)["articles"]
    num_new_files = math.ceil(total_num_articles / num_artcles_per_file)
    num_str_len = math.ceil(math.log10(num_new_files))
    num_str_template = "{{num:0{}d}}".format(num_str_len)
    print(num_str_template)

    def num_str(num: int) -> str:
        return num_str_template.format(num=num)

    num = 0
    new_fcontent = []
    for _, fcontent in dataloader:
        for article in fcontent:
            if len(new_fcontent) == num_artcles_per_file:
                yield fpath_template.format(num=num_str(num)), new_fcontent
                num += 1
                new_fcontent = list()
            new_fcontent.append(article)
    # Yield last if not empty
    if new_fcontent:
        yield fpath_template.format(num=num_str(num)), new_fcontent
