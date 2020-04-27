"""Split or unite a dataset."""

import itertools
import logging
import math

from uqa import dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def unite_dl(data_it: dataset.DataIterable, output_fpath: str) -> dataset.DataIterable:
    """Merge all `data_it` data into a single data object.

    Parameters
    ----------
    data_it: :obj:`.dataset.DataIterble`
        A dateset iterable.
    output_fpath: str
        The path yielded along the merged content.

    Yields
    ------
    str
        `output_fpath`
    :obj:`.dataset.TJson`
        The merged data from `data_it`
    """
    new_fcontent = list()
    for _, data in data_it:
        new_fcontent.extend(data)
    yield output_fpath, new_fcontent


def split_dl(data_it: dataset.DataIterable, fpath_template: str, num_artcles_per_file: int) -> dataset.DataIterable:
    """Split the dataloader data into `num_articles_per_file` articles splits
    and generate new paths with the template `fpath_template` for each split.

    Parameters
    ----------
    dataloader: :obj:`.dataset.Dataloader`
        A dataset iterable
    fpath_template: str
        A bracket style path template string with a '{num}' placeholder.
        '{num}' is formated with the split number automaticlly padded with zeros.
    num_artcles_per_file: int
        The new number of article per file

    Yield
    -----
    str
        A new generated path
    :obj:`dataset.TJson`
        A data split
    """
    data_it, data_it_tee = itertools.tee(data_it)
    total_num_articles = sum(len(fcontent) for _, fcontent in data_it_tee)
    num_new_files = math.ceil(total_num_articles / num_artcles_per_file)
    num_str_len = math.ceil(math.log10(num_new_files))
    num_str_template = "{{num:0{}d}}".format(num_str_len)
    print(num_str_template)

    def num_str(num: int) -> str:
        return num_str_template.format(num=num)

    num = 0
    new_fcontent = []
    for _, fcontent in data_it:
        for article in fcontent:
            if len(new_fcontent) == num_artcles_per_file:
                yield fpath_template.format(num=num_str(num)), new_fcontent
                num += 1
                new_fcontent = list()
            new_fcontent.append(article)
    # Yield last if not empty
    if new_fcontent:
        yield fpath_template.format(num=num_str(num)), new_fcontent
